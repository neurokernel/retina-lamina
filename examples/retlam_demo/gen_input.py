from __future__ import division

import atexit
import os

import numpy as np
import pycuda.driver as cuda

import neurokernel.LPU.utils.simpleio as sio

import retina.retina as ret
import retina.geometry.hexagon as hx
import retina.classmapper as cls_map
from retina.screen.map.mapimpl import AlbersProjectionMap


def gen_input(config):
    cuda.init()
    ctx = cuda.Device(0).make_context()
    atexit.register(ctx.pop)

    suffix = config['General']['file_suffix']

    eye_num = config['General']['eye_num']

    rings = config['Retina']['rings']
    steps = config['General']['steps']

    input_filename = config['Retina']['input_file']
    
    screen_write_step = config['Retina']['screen_write_step']
    config['Retina']['screen_write_step'] = 1
    
    screen_type = config['Retina']['screentype']
    screen_cls = cls_map.get_screen_cls(screen_type)
    eulerangles = config['Retina']['eulerangles']
    radius = config['Retina']['radius']

    for i in range(eye_num):
        screen = screen_cls(config)
        screen_file = 'intensities_tmp{}.h5'.format(i)
        screen.setup_file(screen_file)
    
        retina_elev_file = 'retina_elev{}.h5'.format(i)
        retina_azim_file = 'retina_azim{}.h5'.format(i)

        screen_dima_file = 'grid_dima{}.h5'.format(i)
        screen_dimb_file = 'grid_dimb{}.h5'.format(i)

        retina_dima_file = 'retina_dima{}.h5'.format(i)
        retina_dimb_file = 'retina_dimb{}.h5'.format(i)

        input_file = '{}{}{}.h5'.format(input_filename, i, suffix)

        transform = AlbersProjectionMap(radius,
                                        eulerangles[3*i:3*(i+1)]).invmap
        hexagon = hx.HexagonArray(num_rings=rings, radius=radius,
                                  transform=transform)
        retina = ret.RetinaArray(hexagon, config=config)
        print('Acceptance angle: {}'.format(retina.acceptance_angle))
        print('Neurons: {}'.format(retina.num_photoreceptors))

        elev_v, azim_v = retina.get_ommatidia_pos()

        rfs = _get_receptive_fields(retina, screen, screen_type)
        steps_count = steps
        write_mode = 'w'
        while (steps_count > 0):
            steps_batch = min(100, steps_count)
            im = screen.get_screen_intensity_steps(steps_batch)
            photor_inputs = rfs.filter(im)
            sio.write_array(photor_inputs, filename=input_file, mode=write_mode)
            steps_count -= steps_batch
            write_mode = 'a'
        
        tmp = sio.read_array(screen_file)
        sio.write_array(tmp[::screen_write_step],
                        'intensities{}{}.h5'.format(suffix, i),
                        complevel = 9)
        del tmp
        os.remove(screen_file)

        for data, filename in [(elev_v, retina_elev_file),
                               (azim_v, retina_azim_file),
                               (screen.grid[0], screen_dima_file),
                               (screen.grid[1], screen_dimb_file),
                               (rfs.refa, retina_dima_file),
                               (rfs.refb, retina_dimb_file)]:
            sio.write_array(data, filename)


def _get_receptive_fields(retina, screen, screen_type):
    mapdr_cls = cls_map.get_mapdr_cls(screen_type)
    projection_map = mapdr_cls.from_retina_screen(retina, screen)

    rf_params = projection_map.map(*retina.get_all_photoreceptors_dir())
    if np.isnan(np.sum(rf_params)):
        print('Warning, Nan entry in array of receptive field centers')
    vrf_cls = cls_map.get_vrf_cls(screen_type)
    rfs = vrf_cls(screen.grid)
    rfs.load_parameters(refa=rf_params[0], refb=rf_params[1],
                        acceptance_angle=retina.get_angle(),
                        radius=screen.radius)
    return rfs


def main():
    # TODO read configuration and call function
    # for input generation
    pass

if __name__ == '__main__':
    main()

