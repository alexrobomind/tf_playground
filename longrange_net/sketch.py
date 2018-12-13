class TimeCache:
	def __init__(self, horizon = 5, pre_alignment = 5):
		self._cache = None;
		self._cache_start = tf.Variable(initial_value = 0); # 2 Int tensors holding the [start, stop) interval
		
		self.horizon = horizon;
		self.alignment = alignment;
		
	def __call__(self, data, t_start, t_end):
		# If this is the first call, initialize the cache to be compatible with the passed data
		if not self._cache:
			data_shape = tf.shape(data);
			self._cache = tf.Variable(initial_value = tf.zeros(shape = (data_shape[0], self.horizon, data_shape[1]); # Variable holding the cache
		
		# Calculate the start time required to meet alignment requirement
		t_start_min = (t_start / self.alignment) * self.alignment; # Note: Integer division rounds down
		
		assertions = [
			tf.Assert(
				tf.logical_or(
					# Either there is no need to read data from cache
					t_start_min == t_start,
					# Or the difference lies completely in the cache
					tf.logical_and(t_start_min >= self._cache_start, t_start <= self._cache_start + tf.shape(self._cache)[1])
				),
				["Meeting alignment requires data no longer present in cache (t_start_aligned, cache_start)", t_start_min, self._cache_start],
				name='assert_tstart'
			)
		];
		
		with tf.control_dependencies(assertions):
			t_start_min = tf.Identity(t_start_min, name = 'do_asserts');
		
		cache_offset      = t_start_min - self._cache_start;
		cache_extract_len = t_start - t_start_min;
		
		@autograph.convert
		def inner():