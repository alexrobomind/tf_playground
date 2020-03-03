import tensorflow as tf

# Attention layers

def masked_softmax(logits, axis = -1, mask = None):
    if mask is None:
        return tf.nn.softmax(logits, axis = axis)
    
    logits -= tf.gradient_stop(tf.math.reduce_max(logits, axis = axis, keepdims = True))
    weights = tf.exp(logits)
    weights = tf.where(mask, weights, 0)
    
    norm = tf.math.reduce_sum(weights, axis = axis, keepdims = True)
    norm = tf.where(norm > 0, norm, 1)
    
    weights /= norm
    
    return weights

def full_attention(keys, values, queries, mask_kv = None, mask_q = None):
    """
        keys    - batch_shape + [n_sequence1, dim_k] tensor holding query keys
        values  - batch_shape + [n_sequence1, dim_v] tensor holding query results
        queries - batch_shape + [n_sequence2, dim_k] tensor holding queries
        mask_kv - batch_shape + [n_sequence1] tensor holding the mask for the keys / values (0 if)
        mask_q  - batch_shape + [n_sequence2] tensor holding the mask for the 

        Returns batch_shape + [n_sequence2, dim_v] tensor holding result
    """
    with tf.name_scope('scaled_dot_product_attention'):
        # Compute the [..., n_sequence2, n_sequence1] matrix of weights
        weights = tf.linalg.matmul(queries, keys, transpose_b = True)
        
        # Normalize against the multiplied dimension dimension
        dk = tf.cast(tf.shape(keys)[-1], tf.float32)
        weights = weights / tf.math.sqrt(dk)
        
        if mask_kv is not None or mask_q is not None:
            mask_kv = mask_kv if mask_kv is not None else [True]
            mask_q  = mask_q  if mask_q  is not None else [True]
            
            mask = tf.math.logical_and(
                tf.expand_dims(mask_kv, axis = -2),
                tf.expand_dims(mask_q , axis = -1)
            )
            
            weights = tf.where(mask, weights, tf.math.reduce_min(weights, axis = -1, keepdims = True))
        else:
            mask = None
        
        # Compute softmax over the sequence1 dimension
        weights = masked_softmax(weights, axis = -1, mask = mask)
        
        # Apply weights to values
        #result = tf.linalg.matmul(weights, values)
        result = values
        
        return result


def reduced_attention(keys, values, queries, mask_kv = None, mask_q = None):
    """
        keys    - batch_shape + [n_sequence1, dim_k] tensor holding query keys
        values  - batch_shape + [n_sequence1, dim_v] tensor holding query results
        queries - batch_shape + [n_sequence2, dim_k] tensor holding queries

        Returns batch_shape + [n_sequence2, dim_v] tensor holding result
    """
    with tf.name_scope('reduced_scaled_dot_product_attention'):
        if mask_kv is not None:
            keys = tf.where(mask_kv, keys, 0)
            values = tf.where(mask_kv, values, 0)
        
        if mask_q is not None:
            queries = tf.where(mask_q, queries, 0)
        
        # Compute the [..., dim_k, dim_v] matrix of weights
        #weights = tf.linalg.matmul(keys, values, transpose_a = True)
        weights = tf.eye(tf.shape(keys)[-1], tf.shape(values)[-1])
        
        # Normalize against the key dimension
        if mask_kv is not None:
            dn = tf.reduce_sum(tf.cast(mask_kv, tf.float32), axis = -1)
            
            # We need to expand from a batch_shape to a batch_shape + [1, 1] array
            # to maintain correct broadcasting
            dn = tf.expand_dims(dn, axis = -1)
            dn = tf.expand_dims(dn, axis = -1)
        else:
            dn = tf.cast(tf.shape(keys)[-2], tf.float32)
        
        weights = weights / tf.math.sqrt(dn)
        
        # Compute softmax over the dim_k dimension
        weights = tf.nn.softmax(weights, axis = -2)
        
        # Apply weights to values
        result = tf.linalg.matmul(queries, weights)
        
        return result

class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, d_out, d_model, num_heads, f = reduced_attention, skip = True):
        super(MultiHeadedAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
    
        self.wk = tf.keras.layers.Dense(num_heads * d_model, name = 'Wk', use_bias = True, kernel_initializer = 'glorot_normal')
        self.wv = tf.keras.layers.Dense(num_heads * d_model, name = 'Wv', use_bias = True, kernel_initializer = 'glorot_normal')
        self.wq = tf.keras.layers.Dense(num_heads * d_model, name = 'Wq', use_bias = True, kernel_initializer = 'glorot_normal')
        
        self.wout = tf.keras.layers.Dense(d_out, name = 'Wout', use_bias = True, activation = 'relu', kernel_initializer = 'glorot_normal')
        
        self.f = f
        self.skip = skip
        
    def _split_heads(self, x):
        with tf.name_scope('split_heads/'):
            # Ensure static shape
            #x.set_shape(tf.TensorShape([None, None, num_heads * d_model]))
            #static_shape = x.get_shape();

            # Dynamic reshape
            batch_shape = tf.shape(x)[:-2]
            batch_dims = tf.size(batch_shape)
            
            #batch_size = tf.shape(x)[0]
            seq_len = tf.shape(x)[-2]

            x = tf.reshape(x, tf.concat([
                batch_shape,
                [seq_len, self.num_heads, self.d_model]
            ], axis = 0))

            #x = tf.transpose(x, [0, 2, 1, 3])
            x = tf.transpose(x, tf.concat([
                tf.range(batch_dims),
                [batch_dims + 1, batch_dims + 0, batch_dims + 2]
            ], axis = 0))
        return x
    
    def _merge_heads(self, x):
        with tf.name_scope('merge_heads/'):
            batch_shape = tf.shape(x)[:-3]
            batch_dims  = tf.size(batch_shape)
            
            seq_len = tf.shape(x)[-2]
            
            x = tf.transpose(x, tf.concat([
                tf.range(batch_dims),
                [batch_dims + 1, batch_dims + 0, batch_dims + 2]
            ], axis = 0))
            
            x = tf.reshape(x, tf.concat([
                batch_shape,
                [seq_len, self.num_heads * self.d_model]
            ], axis = 0))
            
            return x
        
    def call(self, kvq, mask = None):
        if isinstance(kvq, tuple):
            k, v, q = kvq
            
            if mask is not None:
                kv_mask = mask[0]
                q_mask  = mask[2]
        else:
            k = kvq
            v = kvq
            q = kvq
            kv_mask = mask
            q_mask  = mask
        
        if self.skip:
            q2 = q

        k = self.wk(k)
        v = self.wv(v)
        q = self.wq(q)
        
        k = self._split_heads(k)
        v = self._split_heads(v)
        q = self._split_heads(q)
        
        result = self.f(k, v, q)
        result = self._merge_heads(result)
        
        result = self.wout(result)
        
        if self.skip:
            result += q2
        
        return result
    
    def compute_mask(self, kvq, mask = None):
        if isinstance(kvq, tuple) and mask is not None:
            return mask[2]
        
        return mask

# --- Wraps another callable as a layer that accepts packed tensors ---

class StackedTogether(tf.keras.layers.Layer):
    def __init__(self, wrapped, axis = -2, mask_axis = None):
        if mask_axis is None:
            mask_axis = axis
            # TODO: Warn if axis < 0

        super(StackedTogether, self).__init__()
        
        self._supports_ragged_inputs = True
        
        self.wrapped = wrapped
        self.axis = axis
        self.mask_axis = mask_axis
        
    def _compute(self, inputs, mask = None, *args, **kwargs):
        inputs_flat = tf.nest.flatten(inputs)
        sizes = [tf.shape(x)[self.axis] for x in inputs_flat]
        axis = self.axis if self.axis >= 0 else tf.rank(inputs_flat[0]) + self.axis
        
        if mask is not None:
            mask = tf.concat(
                tf.nest.flatten(mask),
                axis = axis
            )
        
        stacked = tf.concat(inputs_flat, axis = self.axis)
        
        out_data = self.wrapped(stacked, *args, **kwargs)
        out_mask = self.wrapped.compute_mask(stacked, *args, **kwargs)
        
        out_data = tf.nest.pack_sequence_as(inputs, tf.split(out_data, sizes, axis = axis))
        out_mask = tf.nest.pack_sequence_as(inputs, tf.split(out_mask, sizes, axis = axis))
        
        return out_data, out_mask
    
    def call(self, inputs, mask = None, *args, **kwargs):
        return self._compute(inputs, mask, *args, **kwargs)[0]
    
    # TODO: This is not really safe to merge into one call, because the arguments required
    # for the wrapped __call__ might not be passed in compute_mask
    def compute_mask(self, inputs, mask = None, *args, **kwargs):
        return self._compute(inputs, mask = None, *args, **kwargs)[1]