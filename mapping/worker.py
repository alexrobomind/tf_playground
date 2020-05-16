import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import optuna
import netCDF4
import io
import time
import sys
import yaml
from tqdm import tqdm, trange

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Load study
study = optuna.load_study(
    storage = 'sqlite:///storate.sqlite',
    study_name = sys.argv[1],
    sampler = optuna.samplers.TPESampler(),
    pruner = optuna.pruners.HyperbandPruner()
)

# Read index of input data
root = study.user_attrs['root']

with open(root + '\\index.yaml') as f:
    index = yaml.load(f)

configs = index['configs']
n_configs = len(configs)

# Prepare storage for point data and parameters
n_fs_max = 1000
n_pc_max = 100 * 100
n_pc_center = 100

xyz = tf.Variable(tf.constant(
    float('nan'),
    dtype = tf.float32,
    shape = [3, n_configs, n_fs_max, n_pc_max]
))
ds  = tf.Variable(tf.constant(
    float('nan'),
    dtype = tf.float32,
    shape = [n_configs, n_fs_max, n_fs_max]
))

param_vars = {
    'i_axis' : tf.float32
}
params = {
    k : tf.Variable(tf.zeros(
        shape = [len(index['configs'])],
        dtype = dtype
    ))
    for k, dtype in param_vars.items()
}

def open_config(i):
    # Python function that loads the file and gets the parameters
    def loader(i_config):
        i_config = i_config.numpy()
        conf = index['configs'][i_config]

        distance_file = root + '\\mininum_distance\\{}.nc'.format(conf['name'])
        pc_file       = root + '\\pc\\{}.nc'.format(conf['name'])
    
        with netCDF4.Dataset(distance_file) as nc:
            inp_ds = tf.constant(nc.variables['ds'][:], tf.float32)

        with netCDF4.Dataset(pc_file) as nc:
            inp_xyz = nc.variables['data'][:]

        n_fs = tf.reduce_prod(tf.shape(inp_xyz)[1:3])
        n_pc = tf.reduce_prod(tf.shape(inp_xyz)[3:])
        inp_xyz  = tf.reshape(inp_xyz, [3, n_fs, n_pc])
        inp_xyz  = tf.transpose(inp_xyz, [1, 2, 0])
        
        params_flat = [
            tf.constant(conf[k], dtype = tf.float32)
            for k in params
        ]
        
        return [inp_xyz, inp_ds] + params_flat
    
    loader_output = tf.py_function(
        loader,
        i,
        [tf.float32] * (2 + len(params))
    )
    
    inp_xyz = loader_output[0]
    inp_ds  = loader_output[1]
    params_flat = loader_output[2:]
    
    return tf.Dataset.from_tensors({
        'xyz' : inp_xyz,
        'ds'  : inp_ds,
        'params' : {
            k : params_flat[i] for i, k in enumerate(params)
        }
    })

def sample(config, samples_per_block):
    xyz = config['xyz']
    ds  = config['ds']
    
    n_fs = tf.shape(xyz)[1]
    n_pc = tf.shape(xyz)[2]
    
    def make_idx(n_i):
        return tf.random.uniform(shape = [samples_per_block], minval = 0, maxval = n_i, dtype = tf.int32)
    
    i1 = make_idx(n_fs)
    i2 = make_idx(n_fs)
    
    j1 = make_idx(n_pc)
    j2 = make_idx(n_pc)
    
    xyz1 = tf.gather_nd(
        xyz,
        tf.stack([i1, j1], axis = -1)
    )
    xyz2 = tf.gather_nd(
        xyz,
        tf.stack([i2, j2], axis = -1)
    )
    d = tf.gather_nd(
        ds,
        tf.stack([i1, i2], axis = -1)
    )
    
    ic = tf.zeros(dtype = tf.int64, shape = [samples_per_block])
    jc = make_idx(n_pc)
    
    xyzc = tf.gather_nd(
        xyz,
        tf.stack([ic, jc])
    )
    
    return {
        'xyz1' : xyz1,
        'xyz2' : xyz2,
        'xyzc' : xyzc,
        'd' : d,
        'params' : config['params']
    }

def merge_blocks(input):
    shape = tf.shape(input)
    
    new_shape = tf.concat([shape[0] * shape[1], shape[2:]], axis = 1)
    
    return tf.reshape(
        input,
        new_shape
    )

def select_indices(purpose):
    indices = [k for k, config in enumerate(index['configs']) if config['purpose'] == purpose]
    assert len(indices) > 0, 'No configurations with purpose {} found'.format(purpose)
    
    return indices

n_concurrent_files = 5
blocks_per_file = 50
blocks_per_batch = 5
samples_per_block = 20000

def training_data(purpose):
    return (
        tf.Dataset.from_tensor(select_indices('training'))
        .repeat()
        .shuffle(10)
        .interleave(lambda x: open_config(x).repeat(blocks_per_file), cycle_length = n_concurrent_files)
        .map(lambda x: sample(x, samples_per_block))
        .batch(blocks_per_batch)
        .map(merge_blocks)
        .prefetch(10)
    )

def validation_data():
    indices = select_indices('validation')
    
    if len(indices) > 10:
        indices = indices[:10]
        
    return (
        tf.Dataset.from_tensor(indices)
        .flat_map(open_config)
        .map(lambda x: sample(x, samples_per_block))
        .batch(len(indices))
        .map(merge_blocks)
        .cache('validation_cache')
        .repeat()
    )

# Load point data and variables
for i_config, config in enumerate(tqdm(index['configs'], desc = 'Loading input data')):
    distance_file = root + '\\mininum_distance\\' + config['name'] + '.nc'
    pc_file       = root + '\\pc\\'               + config['name'] + '.nc'
    
    with netCDF4.Dataset(distance_file) as nc:
        inp_ds = tf.constant(nc.variables['ds'][:], tf.float32)

    with netCDF4.Dataset(pc_file) as nc:
        inp_xyz = nc.variables['data'][:]

    n_fs = tf.reduce_prod(tf.shape(inp_xyz)[1:3])
    n_pc = tf.reduce_prod(tf.shape(inp_xyz)[3:])
    
    #tqdm.write(str(n_fs.numpy()))
    #tqdm.write(str(n_pc.numpy()))
    #tqdm.write(str(tf.shape(inp_xyz).numpy()))
    
    inp_xyz  = tf.reshape(inp_xyz, [3, n_fs, n_pc])
    
    n_fs = tf.minimum(n_fs, n_fs_max)
    n_pc = tf.minimum(n_pc, n_pc_max)
    
    xyz[:, i_config, :n_fs, :n_pc].assign(
        inp_xyz[:, :n_fs, :n_pc]
    )
    
    ds[i_config, :n_fs, :n_fs].assign(
        inp_ds[:n_fs, :n_fs]
    )
    
    for k in params:
        params[k][i_config].assign(config[k])

# Freeze variables into constants
xyz = tf.identity(xyz)
ds  = tf.identity(ds)
params = {
    k : tf.identity(v) for k, v in params.items()
}

# Compute indices for training and validation
idx_train = tf.constant(
    [i for i, x in enumerate(index['configs']) if x['purpose'] == 'training'],
    dtype = tf.int32
)

idx_validate = tf.constant(
    [i for i, x in enumerate(index['configs']) if x['purpose'] == 'validation']
)

print(tf.shape(xyz))
print(tf.shape(ds))
print(params)

# Split input data into training and validation set
# Take % of the Poincare surfaces for validation
#d = study.user_attrs['validation_split']#xyz.shape[3] // study.user_attrs['validation_split']
#mask_train = [True if i % d != 0 else False for i in range(xyz.shape[3])]
#mask_val   = [not x for x in mask_train]
#
#tqdm.write(str(mask_train))
#tqdm.write(str(mask_val))
#
#xyz_val = xyz[:,:,:,mask_val,:]
#xyz     = xyz[:,:,:,mask_train,:]
#
#xyz = tf.constant(xyz, dtype = tf.float32)
#xyz_val = tf.constant(xyz_val, dtype = tf.float32)
#
#tqdm.write(str(xyz.shape))
#tqdm.write(str(xyz_val.shape))
#assert ds.shape == (n_fs, n_fs), 'Distance file shape mismatch'
#
#xyz = tf.reshape(xyz, [3, n_fs, -1])
#xyz_val = tf.reshape(xyz_val, [3, n_fs, -1])

# Batch generation
def make_batch(n, validation = False):
    # Choose correct data source
    #src = xyz_val if validation else xyz
    idx = idx_validate if validation else idx_train
    
    # Make random indices into source to gather pairs
    def midx(n_i):
        return tf.random.uniform(shape = [n], minval = 0, maxval = n_i, dtype = tf.int32)
    
    i1 = midx(tf.shape(xyz)[2])
    i2 = midx(tf.shape(xyz)[2])
    
    j1 = midx(tf.shape(xyz)[3])
    j2 = midx(tf.shape(xyz)[3])
    
    k = midx(tf.size(idx))
    k = tf.gather(idx, k)
    
    # Gather pairs and distances
    xyz1 = tf.gather_nd(
        tf.transpose(xyz, [1, 2, 3, 0]),
        tf.stack([k, i1, j1], axis = -1)
    )
    xyz2 = tf.gather_nd(
        tf.transpose(xyz, [1, 2, 3, 0]),
        tf.stack([k, i2, j2], axis = -1)
    )
    d = tf.gather_nd(
        ds,
        tf.stack([k, i1, i2], axis = -1)
    )
    
    # Mask out invalid values
    mask = tf.logical_and(
        xyz1[...,0] == xyz1[...,0],
        xyz2[...,0] == xyz2[...,0]
    )
    
    mask = tf.logical_and(
        mask,
        tf.math.is_finite(d)
    )
    
    xyz1 = tf.boolean_mask(xyz1, mask, axis = 0)
    xyz2 = tf.boolean_mask(xyz2, mask, axis = 0)
    d    = tf.boolean_mask(d, mask, axis = 0)
    
    # Load parameter data and filter apply mask
    def get_param(p):
        par = tf.gather(
            params[p],
            k,
            axis = 0
        )
        par = tf.boolean_mask(par, mask, axis = 0)
        return par
    
    ps = {
        p : get_param(p) for p in params
    }
    
    #tf.print(tf.shape(xyz1))
    #tf.print(tf.shape(xyz2))
    #tf.print(tf.shape(d))
    #
    #for v in ps.values():
    #    tf.print(tf.shape(v))
    
    return xyz1, xyz2, d, ps

# Model class and parameters
class Model(tf.keras.Model):
    def __init__(self, n_classes, n_phi, n_layers = 4, d_layer = 512):
        super().__init__()
        
        self.l = [
            l
            for i in range(n_layers)
            for l in [
                tf.keras.layers.Dense(d_layer, use_bias = True),
                tf.keras.layers.LeakyReLU()
            ]
        ]
        
        self.out_l = tf.keras.layers.Dense(n_classes, use_bias = True)
        self.out_2 = tf.keras.layers.Dense(1, use_bias = False)
        
        self.n_phi = tf.constant(n_phi, dtype = tf.float32)
        x0, y0, z0 = xyz[:,0,0,0]
        
        self.z0 = z0
        self.r0 = tf.sqrt(x0**2 + y0**2)
    
    def call(self, inputs):
        xyz = tf.cast(inputs['xyz'], tf.float32)
        
        def normalized_param(p):
            vals = [c[p] for c in configs]
            
            maxval = max(vals)
            minval = min(vals)
            
            pval = tf.cast(inputs['params'][p], dtype = tf.float32)
            
            return (pval - minval) / max(maxval - minval, 1)
            
        params = [
            normalized_param(p)
            for p in param_vars
        ]
        params = tf.stack(params, axis = -1)
        
        # Pre-processing
        x = xyz[...,0]
        y = xyz[...,1]
        z = xyz[...,2]
        
        r = tf.sqrt(x**2 + y**2)
        p = tf.atan2(y, x)
        
        coss = tf.cos(
            p[...,None] * self.n_phi
        )
        sins = tf.sin(
            p[...,None] * self.n_phi
        )
        
        x = tf.concat(
            [
                tf.tanh(r[...,None] - self.r0),
                tf.tanh(z[...,None] - self.z0),
                sins,
                coss,
                params
            ],
            axis = -1
        )
        
        # Layer application
        for l in self.l:
            x = l(x)
        
        return self.out_l(x), self.out_2(x)[...,0]

# Wrapper that normalizes the model against the axis
class Wrapper(tf.keras.Model):
    def __init__(self, m):
        super().__init__()
        
        self.m  = m
    
    def call(self, d):
        _,   rad0 = self.m(xyz[None,:, 0, 0])
        cls, rad  = self.m(d)
        
        rad = tf.abs(rad - rad0)
        return cls, rad

# Helper that lets us make plots
class SummaryPlot:
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        pass
    
    def __exit__(self, t, v, tb):
        # Encode the current figure as png
        buf = io.BytesIO()
        plt.savefig(buf, format = 'png')
        plt.close()
        buf.seek(0)
        
        # Load png into tf
        img = tf.image.decode_png(buf.getvalue(), channels = 4)
        img = img[None, ...]
        
        # Write as summary
        tf.summary.image(self.name, img)

def center_loss(model):
    # We would prefer points on the center to be mapped to 0
    points = tf.reshape(xyz[:,:,0,:n_pc_center], [3, -1])
    points = tf.transpose(points)
    
    def get_param(p):
        p = p[...,None]
        p = tf.broadcast_to(
            p,
            shape = tf.shape(xyz[0,:,0,:n_pc_center])
        )
        p = tf.reshape(p, [-1])
        return p
        
    c_params = tf.nest.map_structure(
        get_param,
        params
    )
    
    _, coords = model({'xyz' : points, 'params' : c_params})
    
    return tf.reduce_mean(coords**2)

def batch_loss(model, x1, x2, d, xc, params, c_loss_weight = 0, min_scale = 0, summary = False):    
    classes1, coords1 = model({'xyz' : x1, 'params' : params})
    classes2, coords2 = model({'xyz' : x2, 'params' : params})
    
    distances = tf.abs(coords1 - coords2)
    ideal_distances = d
    
    # We want the distances in the same zone to match the real distance, and also couple cross-zone a bit
    distance_loss = tf.abs(
        (distances - ideal_distances)
    )[..., None, None]**2
    distance_loss -= min_scale ** 2
    tf.debugging.assert_all_finite(distance_loss, 'distances')
    
    probs = tf.nn.softmax(classes1)[..., None] * tf.nn.softmax(classes2)[..., None, :]
    distance_loss *= probs
    tf.debugging.assert_all_finite(distance_loss, 'weights')
    
    distance_loss = tf.linalg.trace(distance_loss)# + 1e-2 * tf.reduce_sum(distance_loss, axis = [-2, -1])
    tf.debugging.assert_all_finite(distance_loss, 'reduced')
    
    distance_loss += min_scale ** 2
    
    _, center_coords = model({'xyz' : xc, 'params' : params})
    c_loss = tf.reduce_mean(center_coords ** 2)
    
    #c_loss = center_loss(model)
    
    # We want points that are close together in the same zone
    # The following statemend computes the log probability of two points being in the same zone
    #class_loss = -tf.reduce_logsumexp(
    #    tf.nn.log_softmax(classes1) + tf.nn.log_softmax(classes2), axis = -1
    #)
    #tf.debugging.assert_all_finite(distance_loss, 'classes 1')
    #class_loss *= tf.exp(ideal_distances / -0.01)
    #tf.debugging.assert_all_finite(distance_loss, 'classes 2')
    
    if summary:        
        with SummaryPlot('Class distance histogram (log)'):
            prob_trace = tf.linalg.trace(probs)
            mask       = prob_trace > 0.8
            
            plt.figure(figsize = (6, 6))
                        
            plt.hist2d(
                tf.math.log(ideal_distances[mask] + 1e-4) / tf.math.log(10.0),
                tf.math.log(distances[mask] + 1e-4) / tf.math.log(10.0),
                bins = 50,
                #norm = col.LogNorm()
            )
            plt.axis('equal')
            bar = plt.colorbar()
            bar.set_label('Counts')
            
            plt.plot([-4, 0], [-4, 0], c = 'red')
            plt.title('Intra-Class distance histogram')
            
            plt.xlabel('Log 10 of Actual distance')
            plt.ylabel('Log 10 of Estimated distance')
        
        with SummaryPlot('Intra-Class distance histogram (fine)'):
            prob_trace = tf.linalg.trace(probs)
            mask       = prob_trace > 0.8
            
            plt.figure(figsize = (6, 6))
                        
            plt.hist2d(
                ideal_distances[mask],
                distances[mask],
                bins = 20,
                #norm = col.LogNorm(),
                range = [[0, 0.01], [0, 0.01]]
            )
            plt.axis('equal')
            bar = plt.colorbar()
            bar.set_label('Counts')
            
            plt.plot(
                [0, 0.01],
                [0, 0.01],
                c = 'red'
            )
            plt.title('Intra-Class distance histogram')
            
            plt.xlabel('Actual distance')
            plt.ylabel('Estimated distance')
    
    distance_loss = tf.reduce_mean(distance_loss)
    
    tf.summary.scalar(
        'Distance loss', distance_loss
    )
    
    tf.summary.scalar(
        'Center loss', c_loss
    )
    
    return distance_loss + c_loss_weight * c_loss

def study_function(trial):
    # Clean up TF backend
    tf.keras.backend.clear_session()
    
    def make_param(name, t):
        minval = trial.study.user_attrs['min_' + name]
        maxval = trial.study.user_attrs['max_' + name]
        
        if t == 'int':
            return trial.suggest_int(name, minval, maxval)
        
        if t == 'float':
            return trial.suggest_uniform(name, minval, maxval)
        
        if t == 'log':
            return trial.suggest_loguniform(name, minval, maxval)
    
    learn_rate = make_param('lr', 'log')
    decay      = make_param('lr_decay', 'log')
    
    baselines = study.user_attrs['baselines']
    #trial.set_user_attr('baselines', baselines)
    #baseline  = baselines[
    #    trial.suggest_int('baseline_idx', 0, len(baselines) - 1)
    #]
    baseline = trial.suggest_categorical(
        'baselines', baselines
    )
    
    # Sample model from parameter suggestions or make new one
    if baseline == 'new':
        n_layers = make_param('n_layers', 'int')
        d_layer = make_param('d_layer', 'int')
        n_cls   = make_param('n_cls', 'int')

        n_phi = study.user_attrs['n_phi']

        model = Model(
            n_cls, n_phi, n_layers, d_layer
        )
    else:
        model = tf.keras.models.load_model(baseline)
        
    #wrap = Wrapper(model)
    
    # Get total time we have
    t_max = trial.study.user_attrs['t_max']
    if t_max <= 0:
        raise ValueError('No time allocated')
    
    t = tf.Variable(0, dtype = tf.int32)
    
    # Set up optimizer
    def lr():
        l = t / t_max
        return learn_rate * (1 - l) + learn_rate * decay * l
    
    opt = tf.keras.optimizers.Adam(lr)
    
    batch_size = trial.study.user_attrs['batch_size']
    train_loss = tf.Variable(0, dtype = tf.float32)
    
    @tf.function
    def do_min(it):
        batch = it.get_next()
        input = [
            batch[k]
            for k in ['xyz1', 'xyz2', 'd', 'xyzc', 'params']
        ]
        
        def loss():
            #input = make_batch(batch_size)
            val = batch_loss(
                model,
                *input,
                c_loss_weight = trial.study.user_attrs['center_loss_weight'],
                min_scale     = trial.study.user_attrs['min_scale']
            )
            
            tf.summary.scalar('Training loss', val)

            train_loss.assign(val)
            return val

        opt.minimize(loss, model.trainable_variables)
        return train_loss

    # Set up reporting
    def report():
        # Report validation loss and scatter graph
        input = make_batch(batch_size, validation = True)
        loss = batch_loss(
            model,
            *input,
            c_loss_weight = trial.study.user_attrs['center_loss_weight'],
            min_scale     = trial.study.user_attrs['min_scale'],
            summary = True
        )
        
        tf.summary.scalar('Validation loss', loss)
        
        with SummaryPlot('Mid-plane'):
            fig, ax = plt.subplots(2, 1, sharex = True)
            
            x = np.linspace((5.5, 0.0, 0.0), (6.5, 0.0, 0.0), 200)
            params = {
                'i_axis' : tf.broadcast_to(0, tf.shape(x)[:-1])
            }
            #classes, positions = wrap(x)
            classes, positions = model({
                'xyz' : x,
                'params' : params
            })
            classes = tf.nn.softmax(classes)
            classes = tf.transpose(classes)
            
            plt.sca(ax[0])
            for cls in classes:
                plt.scatter(x[:,0], positions, s = tf.maximum(cls, 0.001))
            plt.ylabel('r_{pred}')
            
            plt.sca(ax[1])
            for cls in classes:
                plt.plot(x[:,0], cls)
            plt.ylabel('p_cls')
            plt.xlabel('R')
        
        def plot_plane(deg):
            with SummaryPlot('Phi {} plane'.format(deg)):
                fig, ax = plt.subplots(1, 2, sharey = True, figsize = (24, 12))

                r = np.linspace(4.0, 7.0, 300)
                z = np.linspace(-1.5, 1.5, 300)
                gr, gz = np.meshgrid(r, z, indexing = 'ij')
                gx = np.cos(np.radians(deg)) * gr
                gy = np.sin(np.radians(deg)) * gr

                vxyz = tf.stack([gx, gy, gz], axis = -1)
                vxyz = tf.reshape(vxyz, [-1, 3])

                #classes, radii = wrap(vxyz)
                params = {
                    'i_axis' : tf.broadcast_to(0, tf.shape(vxyz)[:-1])
                }
                classes, radii = model({
                    'xyz' : vxyz,
                    'params' : params
                })
                classes = tf.nn.softmax(classes, axis = -1)
                classes = tf.reshape(classes, [len(r), len(z), -1])
                i_cls   = tf.argmax(classes, axis = -1)
                radii   = tf.reshape(radii,   [len(r), len(z)])

                plt.sca(ax[0])
                plt.imshow(radii.numpy().transpose(), origin = 'lower')
                plt.colorbar()
                plt.contour(radii.numpy().transpose(), 100, origin = 'lower', colors = 'white', alpha = 0.2, vmin = 0, vmax = 0.5)

                plt.sca(ax[1])
                plt.imshow(i_cls.numpy().transpose(), origin = 'lower', cmap = 'jet', vmin = 0, vmax = classes.numpy().shape[-1] - 1)
                plt.colorbar()
                plt.contour(radii.numpy().transpose(), 100, origin = 'lower', colors = 'white', alpha = 0.5, vmin = 0, vmax = 0.5)
        
        plot_plane(0)
        plot_plane(180)
        
        return loss
    
    # Start summary
    writer = tf.summary.create_file_writer('studies\\{}\\logs\\{}'.format(study.study_name, trial.number))
    
    with writer.as_default():
        step = tf.Variable(0, dtype = tf.int64)
        tf.summary.experimental.set_step(step)
        
        report()
        
        with trange(t_max) as minutes:
            minutes.set_postfix(
                {
                    'id' : trial.number
                }
            )
            
            for minute in minutes:
                t.assign(minute)
                
                with tqdm(total = 60, leave = False) as seconds:
                    m_start = time.time()
                    
                    m_now = time.time()
                    while m_now - m_start < 60:
                        loss = do_min()
                        step.assign_add(1)
                        
                        seconds.update(m_now - m_start - seconds.n)
                        m_now = time.time()
                    
                    seconds.close()
                
                validation_loss = report()
                tqdm.write(str(validation_loss.numpy()))
                trial.report(validation_loss, minute)
                
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
    
    model.save('studies\\{}\\models\\{}'.format(study.study_name, trial.number))
    return validation_loss

study.optimize(
    study_function
)