#!/usr/bin/env python

import os, resource, sys
import argparse

import numpy as np
import networkx as nx

import neurokernel.core_gpu as core
from neurokernel.pattern import Pattern
from neurokernel.tools.logging import setup_logger
from neurokernel.tools.timing import Timer

from retina.LPU import LPU as rLPU
from lamina.LPU import LPU as lLPU
import retina.retina as ret
import lamina.lamina as lam

import retina.geometry.hexagon as r_hx
import lamina.geometry.hexagon as l_hx
from retina.input_generator import RetinaInputGenerator
from retina.screen.map.mapimpl import AlbersProjectionMap
from retina.configreader import ConfigReader

import gen_input as gi

dtype = np.double
RECURSION_LIMIT = 80000


def setup_logging(config):
    '''
        Logging is useful for debugging
        purposes. By default errors that
        are thrown during simulation do
        not appear on screen.
    '''
    log = config['General']['log']
    file_name = None
    screen = False

    if log in ['file', 'both']:
        file_name = 'neurokernel.log'
    if log in ['screen', 'both']:
        screen = True
    logger = setup_logger(file_name=file_name, screen=screen)


def get_retina_id(i):
    return 'retina{}'.format(i)


def get_lamina_id(i):
    return 'lamina{}'.format(i)


def add_retina_LPU(config, i, retina, manager, generator):
    '''
        This method adds Retina LPU and its parameters to the manager
        so that it can be initialized later. Depending on configuration
        input can either be created in advance and read from file or
        generated during simulation by a generator object.

        --
        config: configuration dictionary like object
        i: identifier of eye in case more than one is used
        retina: retina array object required for the generation of
            graph.
        manager: manager object to which LPU will be added
        generator: generator object or None
    '''
    dt = config['General']['dt']
    debug = config['Retina']['debug']
    time_sync = config['Retina']['time_sync']

    input_filename = config['Retina']['input_file']
    output_filename = config['Retina']['output_file']
    gexf_filename = config['Retina']['gexf_file']
    suffix = config['General']['file_suffix']

    if generator is None:
        input_file = '{}{}{}.h5'.format(input_filename, i, suffix)
    else:
        input_file = None

    output_file = '{}{}{}.h5'.format(output_filename, i, suffix)
    gexf_file = '{}{}{}.gexf.gz'.format(gexf_filename, i, suffix)

    # retina also allows a subset of its graph to be taken
    # in case it is needed later to split the retina model to more
    # GPUs
    G = retina.get_worker_nomaster_graph()
    nx.write_gexf(G, gexf_file)

    n_dict_ret, s_dict_ret = rLPU.lpu_parser(gexf_file)
    retina_id = get_retina_id(i)
    modules = []

    manager.add(rLPU, retina_id, dt, n_dict_ret, s_dict_ret,
                input_file=input_file, output_file=output_file,
                device=2*i, debug=debug, time_sync=time_sync,
                modules=modules, input_generator=generator)


def add_lamina_LPU(config, i, lamina, manager):
    '''
        This method adds Lamina LPU and its parameters to the manager
        so that it can be initialized later.

        --
        config: configuration dictionary like object
        i: identifier of eye in case more than one is used
        lamina: lamina array object required for the generation of
            graph.
        manager: manager object to which LPU will be added
        generator: generator object or None
    '''

    output_filename = config['Lamina']['output_file']
    gexf_filename = config['Lamina']['gexf_file']
    suffix = config['General']['file_suffix']

    dt = config['General']['dt']
    debug = config['Lamina']['debug']
    time_sync = config['Lamina']['time_sync']

    output_file = '{}{}{}.h5'.format(output_filename, i, suffix)
    gexf_file = '{}{}{}.gexf.gz'.format(gexf_filename, i, suffix)
    G = lamina.get_graph()
    nx.write_gexf(G, gexf_file)

    n_dict_ret, s_dict_ret = lLPU.lpu_parser(gexf_file)
    lamina_id = get_lamina_id(i)
    modules = []
    manager.add(lLPU, lamina_id, dt, n_dict_ret, s_dict_ret,
                input_file=None, output_file=output_file,
                device=2*i+1, debug=debug, time_sync=time_sync,
                modules=modules, input_generator=None)


def connect_retina_lamina(config, i, retina, lamina, manager):
    '''
        The connections between Retina and Lamina follow
        the neural superposition rule of the fly's compound eye.
        See more information in NeurokernelRFC#2.

        Retina provides an interface to make this connection easier.
        --
        config: configuration dictionary like object
        i: identifier of eye in case more than one is used
        retina: retina array object
        lamina: lamina array object
        manager: manager object to which connection pattern will be added
    '''
    retina_id = get_retina_id(i)
    lamina_id = get_lamina_id(i)
    print('Connecting {} and {}'.format(retina_id, lamina_id))

    retina_selectors = retina.get_all_selectors()
    lamina_selectors = lamina.get_all_selectors()
    with Timer('creation of Pattern object'):
        from_list = []
        to_list = []

        # accounts neural superposition
        rulemap = retina.rulemap
        for ret_sel in retina_selectors:
            # format should be '/ret/<ommid>/<neuronname>'
            _, lpu, ommid, n_name = ret_sel.split('/')
            # find neighbor of neural superposition
            neighborid = rulemap.neighbor_for_photor(int(ommid), n_name)
            # format should be '/lam/<cartid>/<neuronname>'
            lam_sel = lamina.get_selector(neighborid, n_name)

            # setup connection from retina to lamina
            from_list.append(ret_sel)
            to_list.append(lam_sel)

        pattern = Pattern.from_concat(','.join(retina_selectors),
                                      ','.join(lamina_selectors),
                                      from_sel=','.join(from_list),
                                      to_sel=','.join(to_list),
                                      gpot_sel=','.join(from_list+to_list))
        nx.write_gexf(pattern.to_graph(), retina_id+'_'+lamina_id+'.gexf.gz',
                      prettyprint=True)

    with Timer('update of connections in Manager'):
        manager.connect(retina_id, lamina_id, pattern, compat_check=False)


def start_simulation(config, manager):
    steps = config['General']['steps']
    with Timer('retina and lamina simulation'):
        manager.spawn()
        manager.start(steps=steps)
        manager.wait()


def change_config(config, index):
    '''
        Useful if one wants to run the same simulation
        with a few parameters changing based on index value

        Need to modify else part

        Parameters
        ----------
        config: configuration object
        index: simulation index
    '''
    if index < 0:
        pass
    else:
        suffixes = ['__{}'.format(i) for i in range(3)]
        values = [5e-4, 1e-3, 2e-3]

        index %= len(values)
        config['General']['file_suffix'] = suffixes[index]
        config['General']['dt'] = values[index]


def get_input_gen(config):
    '''
        Depending on configuration input can either be created
        in advance and read from file or
        generated during simulation by a generator object.
    '''

    inputmethod = config['Retina']['inputmethod']

    if inputmethod == 'read':
        print('Generating input files')
        with Timer('input generation'):
            gi.gen_input(config)
        return None
    else:
        print('Using input generating function')
        return RetinaInputGenerator(config)


def get_config_obj(args):
    '''
        Gets the configuration object that reads and
        validates inputs. If configuration is invalid
        this function will print relevant errors.
        Because of the specification file, not only
        are the parameters initialized to defaults even
        if they are not specified but they are also converted
        to the correct type e.g int, float, list.
    '''
    conf_name = args.config

    # append file extension if not exist
    conf_filename = conf_name if '.' in conf_name else ''.join(
        [conf_name, '.cfg'])
    conf_specname = os.path.join('..', 'template_spec.cfg')

    return ConfigReader(conf_filename, conf_specname)


def main():
    import neurokernel.mpi_relaunch

    # default limit is low for pickling
    # the data structures passed through mpi
    sys.setrecursionlimit(RECURSION_LIMIT)
    resource.setrlimit(resource.RLIMIT_STACK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='default',
                        help='configuration file')
    parser.add_argument('-v', '--value', type=int, default=-1,
                        help='Value that can overwrite configuration '
                             'by changing this script accordingly. '
                             'It is useful when need to run this script '
                             'repeatedly for different configuration')

    args = parser.parse_args()

    with Timer('getting configuration'):
        conf_obj = get_config_obj(args)
        config = conf_obj.conf
        change_config(config, args.value)

    setup_logging(config)

    eye_num = config['General']['eye_num']
    num_rings = config['Retina']['rings']
    eulerangles = config['Retina']['eulerangles']
    radius = config['Retina']['radius']

    generator = get_input_gen(config)

    manager = core.Manager()
    for i in range(eye_num):
        with Timer('instantiation of retina and lamina #{}'.format(i)):
            transform = AlbersProjectionMap(radius,
                                            eulerangles[3*i:3*(i+1)]).invmap
            r_hexagon = r_hx.HexagonArray(num_rings=num_rings, radius=radius,
                                          transform=transform)
            l_hexagon = l_hx.HexagonArray(num_rings=num_rings, radius=radius,
                                          transform=transform)


            retina = ret.RetinaArray(r_hexagon, config)
            lamina = lam.LaminaArray(l_hexagon, config)

            if generator is not None:
                generator.retina = retina

            add_retina_LPU(config, i, retina, manager, generator)
            add_lamina_LPU(config, i, lamina, manager)

            connect_retina_lamina(config, i, retina, lamina, manager)

    start_simulation(config, manager)


if __name__ == '__main__':
    main()

