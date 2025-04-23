# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

def load_general_tokenizer(text):
    """
    Builds a tokenizer using SubwordTextEncoder for the given text.

    Args:
        text (str): Input text used to generate the vocabulary.

    Returns:
        SubwordTextEncoder: A tokenizer capable of encoding and decoding text.
    """
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus([text], target_vocab_size=2**13)
    return tokenizer


def prepare_decoder_only_data(text, tokenizer, max_tokens, buffer_size=10000, batch_size=64):
    """
    Prepares tokenized data for a decoder-only language model.

    Args:
        text (str): Input text to be tokenized.
        tokenizer (SubwordTextEncoder): Tokenizer used to encode the text.
        max_tokens (int): Maximum sequence length for training.
        buffer_size (int, optional): Size for shuffling the dataset (default: 10,000).
        batch_size (int, optional): Number of sequences per batch (default: 64).

    Returns:
        tf.data.Dataset: A dataset containing tokenized input-output pairs for training.
    """
    tokenized_text = tokenizer.encode(text)
    examples = [tokenized_text[i: i + max_tokens + 1] for i in range(0, len(tokenized_text) - max_tokens, max_tokens)]

    def split_input_output(seq):
        return seq[:-1], seq[1:]

    dataset = tf.data.Dataset.from_tensor_slices(examples)
    dataset = dataset.map(split_input_output)
    dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size=batch_size, drop_remainder=True)

    return dataset


def positional_encoding(length, depth):
    """
    Generates positional encoding for sequence-based models.

    Args:
        length (int): Sequence length.
        depth (int): Embedding dimension.

    Returns:
        tf.Tensor: Positional encoding tensor of shape (length, depth).
    """
    depth /= 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Embedding layer with positional encoding for sequence-based models.

    Args:
        vocab_size (int): Number of unique tokens.
        output_dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, output_dim, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=output_dim)

    def compute_mask(self, *args, **kwargs):
        """
        Computes mask for sequences with padding.
        """
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        """
        Applies embedding and positional encoding.

        Args:
            x (tf.Tensor): Input tensor of token indices.

        Returns:
            tf.Tensor: Embedded input with positional encoding.
        """
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.output_dim, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

    def process(self, x):
        """
        Processes input using embedding and positional encoding.
        """
        return self.call(x)


class BaseAttention(tf.keras.layers.Layer):
    """
    Base attention layer using multi-head attention.

    Args:
        emb_dim (int): Embedding dimension.
        num_heads (int, optional): Number of attention heads (default: 1).
        **kwargs: Additional arguments for MultiHeadAttention.
    """
    def __init__(self, emb_dim, num_heads=1, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=emb_dim, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class MaskedMultiHeadAttention(BaseAttention):
    """
    Multi-head attention with a causal mask for autoregressive models.

    Args:
        x (tf.Tensor): Input tensor.
        context (optional): Context for attention (not used in this implementation).

    Returns:
        tf.Tensor: Processed output after attention and normalization.
    """
    def call(self, x, context=None):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

    def process(self, x):
        return self.call(x)


class FeedForward(tf.keras.layers.Layer):
    """
    Feed-forward neural network with dropout and normalization.

    Args:
        output_dim (int): Output dimension.
        hidden_layer_dim (int): Hidden layer size.
        dropout_rate (float, optional): Dropout rate (default: 0.1).
    """
    def __init__(self, output_dim, hidden_layer_dim, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_layer_dim, activation='relu'),
            tf.keras.layers.Dense(output_dim),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x

    def process(self, x):
        return self.call(x)


class BasicDecoderLayer(tf.keras.layers.Layer):
    """
    A single decoder layer using masked multi-head attention and feed-forward processing.

    Args:
        masked_mha (tf.keras.layers.Layer): Pre-initialized masked multi-head attention layer.
        ffn (tf.keras.layers.Layer): Pre-initialized feed-forward layer.
    """
    def __init__(self, *, masked_mha, ffn):
        super().__init__()
        self.masked_mha = masked_mha
        self.ffn = ffn

    def call(self, x):
        """
        Applies masked self-attention followed by feed-forward processing.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Processed tensor after attention and feed-forward layers.
        """
        x = self.masked_mha(x=x)
        return self.ffn(x)


class BasicDecoder(tf.keras.layers.Layer):
    """
    A transformer-based decoder with positional encoding, multi-head attention, and feed-forward layers.

    Args:
        num_layers (int, optional): Number of decoder layers (default: 1).
        pos_embedding (tf.keras.layers.Layer): Pre-initialized positional embedding layer.
        masked_mha (tf.keras.layers.Layer): Pre-initialized masked multi-head attention layer.
        ffn (tf.keras.layers.Layer): Pre-initialized feed-forward layer.
        dropout_rate (float, optional): Dropout rate (default: 0.1).
    """
    def __init__(self, *, num_layers=1, pos_embedding, masked_mha, ffn, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.pos_embedding = pos_embedding
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [BasicDecoderLayer(masked_mha=masked_mha, ffn=ffn) for _ in range(num_layers)]

    def call(self, x):
        """
        Applies positional encoding, dropout, and sequential decoding layers.

        Args:
            x (tf.Tensor): Input token IDs.

        Returns:
            tf.Tensor: Processed decoder output.
        """
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for layer in self.dec_layers:
            x = layer(x)

        return x

    def process(self, x):
        """
        Processes input using the decoder layers.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Processed output.
        """
        return self.call(x)


class DecoderOnlyTransformer(tf.keras.Model):
    """
    Transformer model for decoder-only architectures.

    Args:
        basic_decoder (tf.keras.layers.Layer): Pre-initialized decoder layer.
        output_vocab_size (int): Size of the output vocabulary.
    """
    def __init__(self, *, basic_decoder, output_vocab_size):
        super().__init__()
        self.basic_decoder = basic_decoder
        self.final_layer = tf.keras.layers.Dense(output_vocab_size)

    def call(self, x):
        """
        Applies the decoder and final linear projection.

        Args:
            x (tf.Tensor): Input token IDs.

        Returns:
            tf.Tensor: Logits for each token in the output vocabulary.
        """
        x = self.basic_decoder(x)
        logits = self.final_layer(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits

    def process(self, x):
        return self.call(x)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def decoder_only_loss_function(y_true, y_pred):
    """
    Computes the categorical cross-entropy loss for decoder-only models.

    Args:
        y_true (tf.Tensor): True token labels.
        y_pred (tf.Tensor): Predicted logits.

    Returns:
        tf.Tensor: Computed loss value.
    """
    return loss_object(y_true, y_pred)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate schedule following inverse square-root decay.

    Args:
        output_dim (int): Embedding dimension.
        warmup_steps (int, optional): Number of warmup steps (default: 4000).
    """
    def __init__(self, output_dim, warmup_steps=4000):
        super().__init__()
        self.output_dim = tf.cast(output_dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        Computes learning rate based on training step.

        Args:
            step (int): Current training step.

        Returns:
            tf.Tensor: Computed learning rate.
        """
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.output_dim) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(output_dim=256)


transformer_optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)


def generate_text(model, tokenizer, seed, num_tokens):
    """
    Generates text using a trained model by iteratively predicting the next token.

    Args:
        model (tf.keras.Model): Pre-trained model for text generation.
        tokenizer: Tokenizer used to encode and decode text.
        seed (str): Initial text seed for generation.
        num_tokens (int): Number of tokens to generate.

    Returns:
        None: Prints the generated text word by word with a delay.
    """
    input_eval = tokenizer.encode(seed)
    input_eval = tf.expand_dims(input_eval, 0)

    for _ in range(num_tokens):
        predictions = model(input_eval)
        predictions = predictions[:, -1, :]
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.concat([input_eval, tf.expand_dims([predicted_id], 0)], axis=-1)

    generated_text = tokenizer.decode(input_eval.numpy().squeeze())

    lines = generated_text.split("\n")
    for line in lines:
        words = line.split()
        for word in words:
            print(word, end=' ', flush=True)
            time.sleep(0.3)
        print()


def load_pt_en_data_100(input_language, output_language):
    """
    Loads a subset of the Portuguese-English translation dataset.

    Args:
        input_language (str): Source language ("pt" or "en").
        output_language (str): Target language ("pt" or "en").

    Returns:
        tf.data.Dataset: A dataset containing 100 translation pairs.
    """
    if input_language == "pt" and output_language == "en":
        reverse = False
    elif input_language == "en" and output_language == "pt":
        reverse = True
    else:
        raise ValueError(f"Unsupported language combination: {input_language} -> {output_language}")

    examples, _ = tfds.load('ted_hrlr_translate/pt_to_en',
                            with_info=True,
                            as_supervised=True,
                            shuffle_files=False)
    train_examples = examples['train']
    train_examples_100 = train_examples.skip(1).take(100)

    if reverse:
        train_examples_100 = train_examples_100.map(lambda pt, en: (en, pt))

    return train_examples_100


def load_tokenizer(languages=['pt', 'en']):
    """
    Loads specified tokenizers in the order provided.

    Args:
        languages (list): List defining the order of tokenizers to load.
                          Options: 'pt' for Portuguese, 'en' for English.

    Returns:
        Tokenizer(s): Single tokenizer or tuple of tokenizers in requested order.
    """
    model_name = 'ted_hrlr_translate_pt_en_converter'

    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir='.', cache_subdir='', extract=True
    )

    tokenizers = tf.saved_model.load('ted_hrlr_translate_pt_en_converter_extracted/ted_hrlr_translate_pt_en_converter')

    results = []
    for lang in languages:
        if lang == 'pt':
            results.append(tokenizers.pt)
        elif lang == 'en':
            results.append(tokenizers.en)
        else:
            raise ValueError(f"Invalid language '{lang}' provided. Use 'pt' or 'en'.")

    return results[0] if len(results) == 1 else tuple(results)


def prepare_batch(inputs, labels, input_tokenizer, output_tokenizer, max_tokens):
    """
    Tokenizes and processes input and label data for encoder-decoder models.

    Args:
        inputs (tf.Tensor or list): Input sequences in text form.
        labels (tf.Tensor or list): Corresponding output sequences in text form.
        input_tokenizer: Tokenizer for encoding the input sequences.
        output_tokenizer: Tokenizer for encoding the output sequences.
        max_tokens (int): Maximum number of tokens allowed per sequence.

    Returns:
        tuple: Processed input and output tensors for training.
    """
    inputs = input_tokenizer.tokenize(inputs)
    inputs = inputs[:, :max_tokens]
    inputs = inputs.to_tensor()

    labels = output_tokenizer.tokenize(labels)
    labels = labels[:, :(max_tokens + 1)]
    labels_inputs = labels[:, :-1].to_tensor()
    labels_outputs = labels[:, 1:].to_tensor()

    return (inputs, labels_inputs), labels_outputs


def prepare_encoder_decoder_data(dataset, input_tokenizer, output_tokenizer, max_tokens, batch_size=64):
    """
    Prepares batched and tokenized data for training an encoder-decoder model.

    Args:
        dataset (tf.data.Dataset): Dataset containing input-output text pairs.
        input_tokenizer: Tokenizer for encoding input sequences.
        output_tokenizer: Tokenizer for encoding output sequences.
        max_tokens (int): Maximum sequence length for tokenization.
        batch_size (int, optional): Number of samples per batch (default: 64).

    Returns:
        tf.data.Dataset: Prefetched and tokenized dataset ready for training.
    """
    return (
        dataset
        .batch(batch_size)
        .map(lambda inputs, labels: prepare_batch(inputs, labels, input_tokenizer, output_tokenizer, max_tokens),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


class EncoderLayer(tf.keras.layers.Layer):
    """
    Single encoder layer consisting of self-attention and feed-forward processing.

    Args:
        mha (tf.keras.layers.Layer): Pre-initialized multi-head attention layer.
        ffn (tf.keras.layers.Layer): Pre-initialized feed-forward layer.
    """
    def __init__(self, *, mha, ffn):
        super().__init__()
        self.self_attention = mha
        self.ffn = ffn

    def call(self, x):
        """
        Applies self-attention followed by feed-forward transformation.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Processed output tensor.
        """
        x = self.self_attention(x)
        return self.ffn(x)


class Encoder(tf.keras.layers.Layer):
    """
    Transformer-based encoder with positional encoding and stacked encoder layers.

    Args:
        pos_embedding (tf.keras.layers.Layer): Pre-initialized positional embedding layer.
        mha (tf.keras.layers.Layer): Pre-initialized multi-head attention layer.
        ffn (tf.keras.layers.Layer): Pre-initialized feed-forward layer.
        num_layers (int, optional): Number of encoder layers (default: 1).
        dropout_rate (float, optional): Dropout rate (default: 0.1).
    """
    def __init__(self, *, pos_embedding, mha, ffn, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.pos_embedding = pos_embedding
        self.enc_layers = [EncoderLayer(mha=mha, ffn=ffn) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        """
        Applies positional encoding, dropout, and stacked encoder layers.

        Args:
            x (tf.Tensor): Input token IDs.

        Returns:
            tf.Tensor: Processed tensor after encoding.
        """
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for layer in self.enc_layers:
            x = layer(x)

        return x

    def process(self, x):
        """
        Processes input using the encoder layers.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Processed output.
        """
        return self.call(x)


class MultiHeadAttention(BaseAttention):
    """
    Multi-head self-attention layer.

    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Processed tensor after applying multi-head attention.
    """
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        return self.layernorm(x)

    def process(self, x):
        return self.call(x)


class CrossAttention(BaseAttention):
    """
    Cross-attention mechanism for transformer models.

    Args:
        x (tf.Tensor): Input tensor.
        context (tf.Tensor): Context tensor used for attention.

    Returns:
        tf.Tensor: Processed output tensor after applying cross-attention.
    """
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x, key=context, value=context, return_attention_scores=True
        )
        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        return self.layernorm(x)

    def process(self, x, context):
        return self.call(x, context)


class DecoderLayer(tf.keras.layers.Layer):
    """
    Transformer decoder layer with masked self-attention, cross-attention, and feed-forward processing.

    Args:
        masked_mha (tf.keras.layers.Layer): Pre-initialized masked multi-head attention layer.
        ca (tf.keras.layers.Layer): Pre-initialized cross-attention layer.
        ffn (tf.keras.layers.Layer): Pre-initialized feed-forward layer.
    """
    def __init__(self, *, masked_mha, ca, ffn):
        super().__init__()
        self.masked_mha = masked_mha
        self.cross_attention = ca
        self.ffn = ffn

    def call(self, x, context):
        """
        Applies masked self-attention, cross-attention, and feed-forward layers.

        Args:
            x (tf.Tensor): Input tensor.
            context (tf.Tensor): Context tensor for cross-attention.

        Returns:
            tf.Tensor: Processed output tensor.
        """
        x = self.masked_mha(x=x, context=context)
        x = self.cross_attention(x=x, context=context)
        self.last_attn_scores = self.cross_attention.last_attn_scores
        return self.ffn(x)


class Decoder(tf.keras.layers.Layer):
    """
    Transformer-based decoder with masked self-attention, cross-attention, and feed-forward layers.

    Args:
        num_layers (int, optional): Number of decoder layers (default: 1).
        pos_embedding (tf.keras.layers.Layer): Pre-initialized positional embedding layer.
        masked_mha (tf.keras.layers.Layer): Pre-initialized masked multi-head attention layer.
        ca (tf.keras.layers.Layer): Pre-initialized cross-attention layer.
        ffn (tf.keras.layers.Layer): Pre-initialized feed-forward layer.
        dropout_rate (float, optional): Dropout rate (default: 0.1).
    """
    def __init__(self, *, num_layers=1, pos_embedding, masked_mha, ca, ffn, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.pos_embedding = pos_embedding
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [DecoderLayer(masked_mha=masked_mha, ca=ca, ffn=ffn) for _ in range(num_layers)]
        self.last_attn_scores = None

    def call(self, x, context):
        """
        Applies positional encoding, dropout, and stacked decoder layers.

        Args:
            x (tf.Tensor): Input token IDs.
            context (tf.Tensor): Context tensor for cross-attention.

        Returns:
            tf.Tensor: Processed output tensor.
        """
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for layer in self.dec_layers:
            x = layer(x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x

    def process(self, x, context):
        """
        Processes input using the decoder layers.

        Args:
            x (tf.Tensor): Input tensor.
            context (tf.Tensor): Context tensor.

        Returns:
            tf.Tensor: Processed output.
        """
        return self.call(x, context)


class EncoderDecoderTransformer(tf.keras.Model):
    """
    Full encoder-decoder transformer model.

    Args:
        encoder (tf.keras.layers.Layer): Pre-initialized encoder layer.
        decoder (tf.keras.layers.Layer): Pre-initialized decoder layer.
        output_vocab_size (int): Size of the output vocabulary.
    """
    def __init__(self, *, encoder, decoder, output_vocab_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.final_layer = tf.keras.layers.Dense(output_vocab_size)

    def call(self, inputs):
        """
        Applies encoder, decoder, and final projection.

        Args:
            inputs (tuple): A tuple containing (context, x).

        Returns:
            tf.Tensor: Logits for each token in the output vocabulary.
        """
        context, x = inputs
        context = self.encoder(context)
        x = self.decoder(x, context)
        logits = self.final_layer(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits

    def process(self, inputs):
        """
        Processes input using the transformer layers.

        Args:
            inputs (tuple): A tuple containing (context, x).

        Returns:
            tf.Tensor: Processed output.
        """
        return self.call(inputs)


def encoder_decoder_loss_function(label, pred):
    """
    Computes masked categorical cross-entropy loss for encoder-decoder models.

    Args:
        label (tf.Tensor): Ground truth token indices.
        pred (tf.Tensor): Predicted token logits.

    Returns:
        tf.Tensor: Normalized loss value considering masked tokens.
    """
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def generate_translation(transformer, input_tokenizer, output_tokenizer, sentence, max_length=128):
    """
    Generates a translation for a given sentence using a transformer model.

    Args:
        transformer (tf.keras.Model): Pre-trained transformer model.
        input_tokenizer: Tokenizer for encoding the input sentence.
        output_tokenizer: Tokenizer for decoding the output sentence.
        sentence (str or tf.Tensor): Sentence to translate.
        max_length (int, optional): Maximum length of the generated translation (default: 128).

    Returns:
        str: Translated text.
    """
    sentence = tf.constant(sentence)

    if len(sentence.shape) == 0:
        sentence = sentence[tf.newaxis]

    sentence = input_tokenizer.tokenize(sentence).to_tensor()
    encoder_input = sentence

    start_end = output_tokenizer.tokenize([''])[0]
    start = start_end[0][tf.newaxis]
    end = start_end[1][tf.newaxis]

    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length):
        output = tf.transpose(output_array.stack())
        predictions = transformer([encoder_input, output], training=False)

        predictions = predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)

        output_array = output_array.write(i + 1, predicted_id[0])

        if predicted_id == end:
            break

    output = tf.transpose(output_array.stack())
    text = output_tokenizer.detokenize(output)[0].numpy().decode("utf-8")

    lines = text.split("\n")
    for line in lines:
        words = line.split()
        for word in words:
            print(word, end=' ', flush=True)
            time.sleep(0.3)
        print()


#Copyright 2022 The TensorFlow Authors.
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Changes have been made to the original code: https://github.com/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb