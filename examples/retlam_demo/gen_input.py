from __future__ import division

import atexit

import numpy as np
import pycuda.driver as cuda

from neurokernel.LPU.utils.simpleio import *

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
    screen_type = config['Retina']['screentype']
    screen_cls = cls_map.get_screen_cls(screen_type)
    eulerangles = config['Retina']['eulerangles']
    radius = config['Retina']['radius']

    screen = screen_cls(config)
    screen.setup_file('intensities{}.h5'.format(suffix))

    write_array(screen.grid[0], 'grid_dima.h5')
    write_array(screen.grid[1], 'grid_dimb.h5')

    for i in range(eye_num):
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
            write_array(photor_inputs, filename=input_file, mode=write_mode)
            steps_count -= steps_batch
            write_mode = 'a'

        for data, filename in [(elev_v, retina_elev_file),
                               (azim_v, retina_azim_file),
                               (screen.grid[0], screen_dima_file),
                               (screen.grid[1], screen_dimb_file),
                               (rfs.refa, retina_dima_file),
                               (rfs.refb, retina_dimb_file)]:
            write_array(data, filename)


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

