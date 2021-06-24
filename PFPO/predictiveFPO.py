import common.loader as ld 
from model.train_bignn import train_bignn
from model.test_bignn import test_bignn
from model.bignn import bignn

#
#
#
if __name__ == '__main__':
    cl_args = ld.parse_args()
    config = ld.ld_config(cl_args['config_path'])

    ds    = ld.ld_pfpo_ds(cl_args['ds_path'], config)   
    model = ld.ld_bignn(config)

    if (cl_args['mode']==0):
        train_bignn(model, ds, config)
    elif (cl_args['mode']==1):
        test_bignn(ds, cl_args['ds_path'], model, config)
    else:
        raise RuntimeError('Unidentified execution mode specified')


