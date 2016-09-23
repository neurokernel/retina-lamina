#!/usr/bin/env python

import os, resource, sys
import argparse

import networkx as nx
import numpy as np

import neurokernel.core_gpu as core
from neurokernel.pattern import Pattern
from neurokernel.tools.logging import setup_logger
from neurokernel.tools.timing import Timer
from neurokernel.LPU.LPU import LPU

import retina.retina as ret
import lamina.lamina as lam
import retina.geometry.hexagon as r_hx
import lamina.geometry.hexagon as l_hx
import gen_input as gi

from retina.InputProcessors.RetinaInputProcessor import RetinaInputProcessor
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
from retina.screen.map.mapimpl import AlbersProjectionMap
from retina.configreader import ConfigReader
from retina.NDComponents.MembraneModels.Photoreceptor import Photoreceptor
from retina.NDComponents.MembraneModels.BufferPhoton import BufferPhoton
from retina.NDComponents.MembraneModels.BufferVoltage import BufferVoltage


dtype = np.double
RECURSION_LIMIT = 80000


def setup_logging(config):
    log = config['General']['log']
    file_name = None
    screen = False

    if log in ['file', 'both']:
        file_name = 'neurokernel.log'
    if log in ['screen', 'both']:
        screen = True
    logger = setup_logger(file_name=file_name, screen=screen)

def get_master_id(i):
    return 'retina{}'.format(i)


def get_worker_id(i):
    return 'retina{}'.format(i+1)

def get_lamina_id(i):
    return 'lamina{}'.format(i)

# number of neurons of `j`th worker out of `worker_num`
# with `total_neurons` neurons overall
def get_worker_num_neurons(j, total_neurons, worker_num):
    num_neurons = (total_neurons-1) // worker_num + 1
    return min(num_neurons, total_neurons - j*num_neurons)


def add_master_LPU(config, retina_index, retina, manager):
    dt = config['General']['dt']
    debug = config['Retina']['debug']
    time_sync = config['Retina']['time_sync']

    input_filename = config['Retina']['input_file']
    output_filename = config['Retina']['output_file']
    gexf_filename = config['Retina']['gexf_file']
    suffix = config['General']['file_suffix']

    output_file = '{}{}{}.h5'.format(output_filename, retina_index, suffix)
    gexf_file = '{}{}{}.gexf.gz'.format(gexf_filename, retina_index, suffix)

    inputmethod = config['Retina']['inputmethod']
    if inputmethod == 'read':
        print('Generating input files')
        with Timer('input generation'):
            input_processor = RetinaFileInputProcessor(config, retina)
    else:
        print('Using input generating function')
        input_processor = RetinaInputProcessor(config, retina)

    input_processor = get_input_gen(config, retina)
    uids_to_record = ['ret_{}_{}'.format(name, i) for i in range(retina.num_elements)
                for name in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']]
    output_processor = FileOutputProcessor([('V',uids_to_record)], output_file, sample_interval=1)

    G = retina.get_master_graph()
    nx.write_gexf(G, gexf_file)

    (comp_dict, conns) = LPU.lpu_parser(gexf_file)
    master_id = get_master_id(retina_index)

    extra_comps = [BufferPhoton, BufferVoltage]

    manager.add(LPU, master_id, dt, comp_dict, conns,
                device = retina_index, input_processors = [input_processor],
                output_processors = [output_processor],
                debug=debug, time_sync=time_sync, extra_comps = extra_comps)


def add_worker_LPU(config, retina_index, retina, manager):
    gexf_filename = config['Retina']['gexf_file']
    suffix = config['General']['file_suffix']

    dt = config['General']['dt']
    debug = config['Retina']['debug']
    time_sync = config['Retina']['time_sync']

    worker_num = config['Retina']['worker_num']
    gexf_file = '{}{}_{}{}.gexf.gz'.format(gexf_filename, 0, retina_index, suffix)

    G = retina.get_worker_graph(retina_index+1, worker_num)
    #G = nx.convert_node_labels_to_integers(G)
    nx.write_gexf(G, gexf_file)

    worker_dev = retina_index

    (comp_dict, conns) = LPU.lpu_parser(gexf_file)
    worker_id = get_worker_id(retina_index)
    
    extra_comps = [Photoreceptor]
    manager.add(LPU, worker_id, dt, comp_dict, conns,
                device=worker_dev, debug=debug, time_sync=time_sync,
                extra_comps = extra_comps)


def add_lamina_LPU(config, lamina_index, lamina, manager):
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

    output_file = '{}{}{}.h5'.format(output_filename, lamina_index, suffix)
    gexf_file = '{}{}{}.gexf.gz'.format(gexf_filename, lamina_index, suffix)
    G = lamina.get_graph()
    nx.write_gexf(G, gexf_file)

    comp_dict, conns = LPU.lpu_parser(gexf_file)
    lamina_id = get_lamina_id(lamina_index)
    
    output_processor = FileOutputProcessor(
                            [('V', None)], output_file,
                            sample_interval=1)

    manager.add(LPU, lamina_id, dt, comp_dict, conns,
                output_processors = [output_processor],
                device=lamina_index+1, debug=debug, time_sync=time_sync)


def connect_master_worker(config, worker_index, retina, manager):
    total_neurons = retina.num_photoreceptors

    worker_num = config['Retina']['worker_num']

    master_id = get_master_id(0)
    worker_id = get_worker_id(worker_index)
    print('Connecting {} and {}'.format(master_id, worker_id))

    with Timer('update of connections in Pattern object'):
        pattern = retina.update_pattern_master_worker(worker_index+1, worker_num)

    with Timer('update of connections in Manager'):
        manager.connect(master_id, worker_id, pattern)


def connect_retina_lamina(config, index, retina, lamina, manager):
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
    retina_id = get_master_id(index)
    lamina_id = get_lamina_id(index)
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
        manager.connect(retina_id, lamina_id, pattern)


def start_simulation(config, manager):
    steps = config['General']['steps']
    with Timer('retina simulation'):
        manager.spawn()
        print('Manager spawned')
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
        suffixes = ['__{}'.format(i) for i in range(4)]
        values = range(1, 5)

        index %= len(values)
        config['General']['file_suffix'] = suffixes[index]
        config['Retina']['worker_num'] = values[index]

def get_input_gen(config, retina):
    inputmethod = config['Retina']['inputmethod']

    if inputmethod == 'read':
        print('Generating input files')
        with Timer('input generation'):
            gi.gen_input(config)
        return None
    else:
        print('Using input generating function')

        return RetinaInputProcessor(config, retina)


def get_config_obj(args):
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

    worker_num = config['Retina']['worker_num']
    num_rings = config['Retina']['rings']
    radius = config['Retina']['radius']
    eulerangles = config['Retina']['eulerangles']

    manager = core.Manager()
    
    with Timer('instantiation of retina and lamina'):
        transform = AlbersProjectionMap(radius, eulerangles).invmap
        r_hexagon = r_hx.HexagonArray(num_rings=num_rings, radius=radius,
                                      transform=transform)
        l_hexagon = l_hx.HexagonArray(num_rings=num_rings, radius=radius,
                                      transform=transform)

        retina = ret.RetinaArray(r_hexagon, config)
        lamina = lam.LaminaArray(l_hexagon, config)
        
        add_master_LPU(config, 0, retina, manager)
        for j in range(worker_num):
            add_worker_LPU(config, j, retina, manager)
            connect_master_worker(config, j, retina, manager)
        
        add_lamina_LPU(config, 0, lamina, manager)

        connect_retina_lamina(config, 0, retina, lamina, manager)

    start_simulation(config, manager)


if __name__ == '__main__':
    main()
