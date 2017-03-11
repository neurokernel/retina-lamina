#!/usr/bin/env python

import os, resource, sys
import argparse

import numpy as np
import networkx as nx

from pyorient.ogm import Graph, Config
import pyorient.ogm.graph

setattr(pyorient.ogm.graph, 'orientdb_version',
        pyorient.ogm.graph.ServerVersion)

import neurokernel.core_gpu as core
from neurokernel.pattern import Pattern
from neurokernel.tools.logging import setup_logger
from neurokernel.tools.timing import Timer
from neurokernel.LPU.LPU import LPU

import neuroarch.models as models
import neuroarch.nk as nk

import retina.retina as ret
import lamina.lamina as lam

import retina.geometry.hexagon as r_hx
import lamina.geometry.hexagon as l_hx
from retina.InputProcessors.RetinaInputIndividual import RetinaInputIndividual
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
from retina.screen.map.mapimpl import AlbersProjectionMap
from retina.configreader import ConfigReader
from retina.NDComponents.MembraneModels.PhotoreceptorModel import PhotoreceptorModel
from retina.NDComponents.MembraneModels.BufferPhoton import BufferPhoton
from retina.NDComponents.MembraneModels.BufferVoltage import BufferVoltage

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


def add_retina_LPU(config, retina_index, manager, graph):
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

    output_file = '{}{}{}.h5'.format(output_filename, retina_index, suffix)
    gexf_file = '{}{}{}.gexf.gz'.format(gexf_filename, retina_index, suffix)
    
    # retina also allows a subset of its graph to be taken
    # in case it is needed later to split the retina model to more
    # GPUs
    node_lpu_0 = graph.LPUs.query(name='retina').one()
    g_lpu_na_0 = node_lpu_0.traverse_owns(max_levels = 2).get_as('nx')
    g_lpu_nk_0 = nk.na_lpu_to_nk_new(g_lpu_na_0)
    prs = [node for node in g_lpu_nk_0.nodes(data=True) \
           if node[1]['class'] == 'PhotoreceptorModel']
    for pr in prs:
        g_lpu_nk_0.node[pr[0]]['num_microvilli'] = 3000
    
    nx.write_gexf(g_lpu_nk_0, gexf_file)

    inputmethod = config['Retina']['inputmethod']
    if inputmethod == 'read':
        print('Generating input files')
        print('Reading retina input from file is not supported yet')
        with Timer('input generation'):
            input_processor = RetinaFileInputProcessor(config, retina)
    else:
        print('Using input generating function')
        input_processor = RetinaInputIndividual(config, prs)

    output_processor = FileOutputProcessor([('V',None)], output_file, sample_interval=1)

    (comp_dict, conns) = LPU.graph_to_dicts(g_lpu_nk_0)
    retina_id = get_retina_id(retina_index)

    extra_comps = [PhotoreceptorModel, BufferPhoton]

    manager.add(LPU, retina_id, dt, comp_dict, conns,
                device = retina_index, input_processors = [input_processor],
                output_processors = [output_processor],
                debug=debug, time_sync=time_sync, extra_comps = extra_comps)


def add_lamina_LPU(config, lamina_index, manager, graph):
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

    node_lpu_0 = graph.LPUs.query(name='lamina').one()
    g_lpu_na_0 = node_lpu_0.traverse_owns(max_levels = 2).get_as('nx')
    g_lpu_nk_0 = nk.na_lpu_to_nk_new(g_lpu_na_0)

    nx.write_gexf(g_lpu_nk_0, gexf_file)
    comp_dict, conns = LPU.graph_to_dicts(g_lpu_nk_0)
    lamina_id = get_lamina_id(lamina_index)
    
    extra_comps = [BufferVoltage]
    
    output_processor = FileOutputProcessor(
                            [('V', None)], output_file,
                            sample_interval=1)

    manager.add(LPU, lamina_id, dt, comp_dict, conns,
                output_processors = [output_processor],
                device=lamina_index+1, debug=debug, time_sync=time_sync,
                extra_comps = extra_comps)


def connect_retina_lamina(config, index, manager, graph):
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
    retina_id = get_retina_id(index)
    lamina_id = get_lamina_id(index)
    print('Connecting {} and {}'.format(retina_id, lamina_id))

    
    node_pat = graph.Patterns.query(name='retina-lamina').one()
    g_pat_na = node_pat.traverse_owns(max_levels = 2).get_as('nx')
    g_pat_nk = nk.na_pat_to_nk(g_pat_na)

    pattern, key_order = Pattern.from_graph(nx.DiGraph(g_pat_nk))

    nx.write_gexf(pattern.to_graph(), retina_id+'_'+lamina_id+'_new.gexf.gz',
                      prettyprint=True)
    
    with Timer('update of connections in Manager'):
        manager.connect(retina_id, lamina_id, pattern,
                        int_0 = key_order.index('retina'),
                        int_1 = key_order.index('lamina'))


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


def get_input_gen(config, retina):
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
        return RetinaInputProcessor(config, retina)


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

    num_rings = config['Retina']['rings']
    eulerangles = config['Retina']['eulerangles']
    radius = config['Retina']['radius']

    graph = Graph(Config.from_url('/retina_lamina', 'admin', 'admin',
                                   initial_drop=False))
    models.create_efficiently(graph, models.Node.registry)
    models.create_efficiently(graph, models.Relationship.registry)

    manager = core.Manager()
    
    with Timer('instantiation of retina and lamina'):
        add_retina_LPU(config, 0, manager, graph)
        add_lamina_LPU(config, 0, manager, graph)

        connect_retina_lamina(config, 0, manager, graph)

    start_simulation(config, manager)


if __name__ == '__main__':
    main()

