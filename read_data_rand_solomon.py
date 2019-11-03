'''
Loads the data in the pre-defined format and creates the variables needed by the model
'''
import math
import os
import sys

from itertools import cycle

import pandas as pd
import numpy as np


####################
# Read excel files
####################

dir_path = os.path.dirname(os.path.realpath(__file__))
data_name = sys.argv[3]
file_path = os.path.join(dir_path, "data", "random", "ELM-TCCUSP-RandGen_"+data_name+".xlsx")
xls = pd.ExcelFile(file_path)

localidade = pd.read_excel(xls, 'Localidade')
capacidades= pd.read_excel(xls, 'Capacidades')
coleta = pd.read_excel(xls, 'Coleta')

########################
# Read Solomon text file
########################

num_instances = int(sys.argv[1])

instance_name = sys.argv[2]+'.txt'
txt_path = os.path.join(dir_path, "data", "random", "In", instance_name)

loc_x = {}
loc_y = {}

with open(txt_path) as f:
    for i, line in enumerate(f):
        if i < 9:
            continue
        l = line.split()
        loc_x[int(l[0])] = int(l[1])
        loc_y[int(l[0])] = int(l[2])
        # we only use the (x, y) coordinates of the points

####################
# Locations
####################

localidade = localidade.append(localidade[localidade['Tipo']=='BASE'].iloc[0], ignore_index=True) # duplicate CD

# service time (unify two columns) - hypothesis: there's only one of them non-zero
localidade['service_time'] = localidade[['Tempo Carga', 'Tempo Descarga']].max(axis=1)

# Daily work and rest shifts and planning horizon
horizon = coleta['Dia'].max()+1
work_shift = 8
rest_shift = 16
day = 24


# number of nodes
# n = len(localidade[localidade['Tipo']=='DESTINO']['Localidade'])
n = len(coleta['Loja'])

##############
# Nodes
##############
# The hypothesis of the model is that multiple deliveries to the same customer are considered different customers

nodes = pd.merge(coleta, localidade, how='left', left_on=['Loja'], right_on=['Localidade'])
nodes = nodes.reset_index()
nodes['index'] = nodes['index'] + 1 # 1-index instead of 0-index, as we will add the deposit
# hypothesis: time starts at 0, aso 8 in the morning of the first day is 0
# nodes['start_window'] = day*(nodes['Dia']-1) + (nodes['inicio'] - start_hour) 
# nodes['end_window'] = day*(nodes['Dia']-1) + (nodes['fim'] - start_hour)
nodes['start_window'] = nodes['inicio']
nodes['end_window'] = nodes['fim']
nodes.drop(columns=['Loja', 'Tipo', 'inicio', 'fim', 'Tempo Carga', 'Tempo Descarga'], inplace=True)
cd_node_0 = {
    'index'         :         [0],
    'M3'            :         [0],
    'Valor (R$)'    :         [0],
    'Localidade'    :      ['CD'],
    'service_time'  :        [localidade[localidade['Localidade']=='CD'].service_time.iloc[0]],
    'start_window'  :        [0],#[localidade[localidade['Localidade']=='CD'].inicio.iloc[0]*0],
    'end_window'    :        [localidade[localidade['Localidade']=='CD'].fim.iloc[0]*horizon],}
cd_node_last = {
    'index'         :         [num_instances+1],
    'M3'            :         [0],
    'Valor (R$)'    :         [0],
    'Localidade'    :      ['CD'],
    'service_time'  :         [0],
    'start_window'  :        [localidade[localidade['Localidade']=='CD'].inicio.iloc[0]*0],
    'end_window'    :        [localidade[localidade['Localidade']=='CD'].fim.iloc[0]*(2*horizon)],} # 2 times the horizon is an arbitrary value, jsut to say that there's no restriction. A tighter bound can be found

nodes = nodes.iloc[0:num_instances].copy(deep=True)
n = num_instances

nodes = pd.concat([pd.DataFrame.from_dict(cd_node_0), nodes, pd.DataFrame.from_dict(cd_node_last)], sort=True)


###########
# Fleet
###########

## Expand fleet
capacidades.loc[capacidades['Veiculos']=='veiculo3', 'Max Qt Veiculos'] = n

capacidades.loc[:, 'Max Qt Veiculos'] = np.minimum(capacidades['Max Qt Veiculos'], num_instances)

K = capacidades['Veiculos'].nunique()
fleetSize = capacidades['Max Qt Veiculos'].sum()

fleet_df = capacidades.reindex(capacidades.index.repeat(capacidades['Max Qt Veiculos'])).reset_index(drop=True).reset_index()


#############################
# Iteration variables Part 1
#############################

customers_list = list(range(1, n+1))
nodes_list = [0] + customers_list + [n+1]
start_nodes_list = nodes_list[:-1]
end_nodes_list = nodes_list[1:]

fleet_list = fleet_df['index'].values
QVolume_list = fleet_df['Capacidades'].values
QValue_list = fleet_df['Maximo Ticket (R$)'].values
fixedCost_list = fleet_df['Custo Fixo'].values
freightCost_list = fleet_df['Frete Km'].values

demand_volume = nodes['M3'].values
demand_value = nodes['Valor (R$)'].values
service_time = nodes['service_time'].values / 60

#############
# Network
#############


E = [[(i, j) for j in range(0, len(loc_x)) if i!=j] for i in range(0, len(loc_x))]
E = [element for sublist in E for element in sublist]

r = 10
speed = 80
dist_dict = {(i, j): r*np.hypot(loc_x[i]-loc_x[j], loc_y[i]-loc_y[j]) for i, j in E}
time_dict = {(i, j): r*np.hypot(loc_x[i]-loc_x[j], loc_y[i]-loc_y[j])/speed for i, j in E}   

malha_arcs = []
for v in capacidades['Veiculos'].values:
    for (i, j) in E:
        record = {'Origem':localidade.iloc[i]['Localidade'],
                  'Destino':localidade.iloc[j]['Localidade'],
                  'Veiculo':v,
                  'Distancia (km)': dist_dict[(i, j)],
                  'Tempo (h)': time_dict[(i, j)]}
        malha_arcs.append(record)

malha = pd.DataFrame.from_records(malha_arcs)


###### TODO: continuar verificando os dados, ver se a malha corresponde ao esperado e rodar heuristica

arcs = malha.copy(deep=True)

# make the arcs take into account the multiplicity of each client
# Source
arcs_dfs = []
for index, name in zip(nodes['index'], nodes['Localidade']):
    if not name=='CD':
        temp  = arcs[arcs['Origem']==name].copy(deep=True)
        temp['start'] = index
        arcs_dfs.append(temp)
temp  = arcs[arcs['Origem']=='CD'].copy(deep=True)
temp['start'] = 0
arcs_dfs.append(temp)
arcs = pd.concat(arcs_dfs, axis=0)
arcs_dfs = []
# Destination
for index, name in zip(nodes['index'], nodes['Localidade']):
    if not name=='CD':
        temp  = arcs[arcs['Destino'] == name].copy(deep=True)
        temp['end']=index
        arcs_dfs.append(temp)
temp  = arcs[arcs['Destino']=='CD'].copy(deep=True)
temp['end']=n+1
arcs_dfs.append(temp)
arcs = pd.concat(arcs_dfs, axis=0)
idle_arcs = {
        'Origem':['CD' for _ in range(K)],
        'Destino':['CD' for _ in range(K)],
        'Veiculo':capacidades['Veiculos'].values,
        'Distancia (km)':[0 for _ in range(K)],
        'Tempo (h)':[0 for _ in range(K)],
        'start':[0 for _ in range(K)],
        'end':[n+1 for _ in range(K)],
 }
arcs = pd.concat([arcs, pd.DataFrame(idle_arcs)], axis=0, sort=True)

#############################
# Iteration variables Part 2
#############################

# hypothesis: if an vehicle is not accepted by a client, there won't be an arc tagged with that vehicle type arriving at or departing from that client

vehicle_acceptance_by_type = {c:[1 for _ in range(K)] for c in nodes_list}
for k in range(K):
    for c in customers_list:
        if not any(arcs[arcs['Veiculo']==capacidades.iloc[k]['Veiculos']].end == c):
            vehicle_acceptance_by_type[c][k] = 0

vehicle_acceptance = {client:[accept for index, accept in enumerate(a_list) for _ in range(capacidades.iloc[index]['Max Qt Veiculos'])] 
                      for client, a_list in vehicle_acceptance_by_type.items()}


##############
# Fix for arcs
##############
# Locations that are visited more than once, we allow them to be serviced again with the same vehicle.

dup_nodes = nodes.loc[nodes['Localidade'].isin(nodes['Localidade'].value_counts()[nodes['Localidade'].value_counts()>1].index.values)&(nodes['Localidade']!='CD'), ['Localidade', 'index']]

self_arcs_dfs = []
for l in dup_nodes['Localidade'].unique():
    temp = dup_nodes[dup_nodes['Localidade']==l]
    self_arcs_df = pd.DataFrame()
    for loc1, ind1 in temp.itertuples(index=False):
        for loc2, ind2 in temp.itertuples(index=False):
            self_arcs = {
                'Origem':[loc1 for k in range(K) if vehicle_acceptance_by_type[ind1][k]],
                'Destino':[loc2 for k in range(K) if vehicle_acceptance_by_type[ind2][k]],
                'Veiculo':[v for k, v in enumerate(capacidades['Veiculos'].values) if vehicle_acceptance_by_type[ind1][k]],
                'Distancia (km)':[0 for k in range(K) if vehicle_acceptance_by_type[ind1][k]],
                'Tempo (h)':[0 for k in range(K) if vehicle_acceptance_by_type[ind1][k]],
                'start':[ind1 for k in range(K) if vehicle_acceptance_by_type[ind1][k]],
                'end':[ind2 for k in range(K) if vehicle_acceptance_by_type[ind2][k]],
             }
            self_arcs_df = pd.concat([self_arcs_df, pd.DataFrame(self_arcs)], axis=0, sort=True)
    self_arcs_dfs.append(self_arcs_df)

if self_arcs_dfs:
    arcs = pd.concat([arcs, pd.concat(self_arcs_dfs, axis=0, sort=True)], axis=0, sort=True)

arcs = arcs[arcs['start']!=arcs['end']]


#############################
# Iteration variables Part 3
#############################

def get_dist(i, j, k):
    vehicle = k if type(k)==str else fleet_df.loc[k]['Veiculos']
    try:

        return arcs[(arcs['Veiculo']==vehicle)&
                (arcs['start']==i)&(arcs['end']==j)][['Distancia (km)']].iloc[0].values[0]
    except:
        return math.inf
def get_time(i, j, k):
    vehicle = k if type(k)==str else fleet_df.loc[k]['Veiculos']
    try:
        return arcs[(arcs['Veiculo']==vehicle)&
                (arcs['start']==i)&(arcs['end']==j)][['Tempo (h)']].iloc[0].values[0]
    except:
        return math.inf
def get_M(i, j, k):
    try:
        return  max(nodes[nodes['index']==i].end_window.iloc[0] + 
                get_time(i, j, k) +
                nodes[nodes['index']==i].service_time.iloc[0] - 
                nodes[nodes['index']==j].start_window.iloc[0], 0)
    except:
        return None
#dist_dict = {(i, j, k): d
#        for k in fleet_list
#        for i, j, d in arcs[arcs['Veiculo']==fleet_df.loc[k]['Veiculos']][['start', 'end', 'Distancia (km)']].itertuples(index=False)}

#time_dict = {(i, j, k): t
#        for k in fleet_list
#        for i, j, t in arcs[arcs['Veiculo']==fleet_df.loc[k]['Veiculos']][['start', 'end', 'Tempo (h)']].itertuples(index=False)}
#M = {(row['start'],row['end'],k_index):      
#     max(nodes[nodes['index']==row['start']].end_window.iloc[0] + row['Tempo (h)'] +
#        nodes[nodes['index']==row['start']].service_time.iloc[0] / 60 - 
#         nodes[nodes['index']==row['end']].start_window.iloc[0], 0)
#     for k_index, vehicle in fleet_df[['index', 'Veiculos']].itertuples(index=False) for index, row in arcs[arcs['Veiculo']==vehicle].iterrows()}

Mp = 100000000
eps = 10e-4

### All vehicles will have a starting time (it will always start a day at that hour)
def compute_early_start_time(clients):
    min_window = nodes[nodes['index'].isin(clients[1:-1])]['end_window'].min()
    ind = nodes[(nodes['index'].isin(clients[1:-1]))&(nodes['end_window']==min_window)]['index'].values[0]
    time_to_min = arcs[(arcs['start']==0)&(arcs['end']==ind)]['Tempo (h)'].max()
    return int(min_window - nodes[nodes['index']==0].service_time.values[0]/60 - time_to_min)


def compute_late_start_time(clients):
    max_window = max(nodes[nodes['index'].isin(clients[1:-1])]['start_window']%24)
    max_service = nodes[(nodes['index'].isin(clients[1:-1]))&(nodes['start_window']%24==max_window)]['service_time'].max()/60
    return math.ceil(max_window + max_service - work_shift)

#min_time = compute_early_start_time(nodes_list)
#max_time = compute_late_start_time(nodes_list)
min_time=6
max_time=7

#min_time = int(min(nodes.iloc[1:-1]['end_window']) - nodes[nodes['index']==0].service_time.values[0]/60 - 
# arcs[(arcs['start']==0)&(arcs['end']==nodes[nodes['end_window']==min(nodes.iloc[1:-1]['end_window'])]['index'].values[0])]['Tempo (h)'].max())
#max_time = math.ceil(max(max(nodes.iloc[1:-1]['start_window']%24) + nodes[nodes['start_window']%24==max(nodes.iloc[1:-1]['start_window']%24)].service_time/60 - 8))

cyc = cycle(range(min_time, max_time+2))
fleet_df['start_hour'] = pd.Series([next(cyc) for k in fleet_df.index], name='start_hour')

vehicle_start_hour = fleet_df['start_hour'].values.astype(np.float)


#def check_feasibility():
#    if len(nodes[(nodes['Dia']==0)&(nodes['index'].isin(arcs[((arcs['Tempo (h)']+nodes[nodes['index']==0].service_time.values[0]/60)>8)&(arcs['start']==0)]['end'].unique()))])>0:
#        print(nodes[(nodes['Dia']==0)&(nodes['index'].isin(arcs[((arcs['Tempo (h)']+nodes[nodes['index']==0].service_time.values[0]/60)>8)&(arcs['start']==0)].end.unique()))])
#        print("Node on day zero cannot be served")
#        return False
#    return True
#
#if not check_feasibility():
#    mod_indexes = nodes[(nodes['Dia']==0)&(nodes['index'].isin(arcs[((arcs['Tempo (h)']+nodes[nodes['index']==0].service_time.values[0]/60)>8)&(arcs['start']==0)].end.unique()))].index
#    if len(mod_indexes)>0:
#        cyc = cycle(range(1, horizon))
#        for i, v in enumerate(mod_indexes):
#            nodes.loc[v, 'Dia'] = next(cyc)
#            nodes.loc[v, 'start_window'] += 24*nodes.loc[v, 'Dia']
#            nodes.loc[v, 'end_window'] += 24*nodes.loc[v, 'Dia']
#        print(nodes.loc[mod_indexes])

mod_indexes = []

for i, row in nodes.iterrows():
    if (row['service_time']/60 + min([get_time(0, row['index'], k) for k in fleet_list]) + nodes[nodes['index']==0].service_time.values[0]/60 > 8
        and row['Dia'] == 0):
        mod_indexes.append(row['index'])

if len(mod_indexes)>0:
    print("These nodes would be infeasible: ",mod_indexes)
    cyc = cycle(range(1, horizon))
    for i, v in enumerate(mod_indexes):
        nodes.loc[nodes['index']==v, 'Dia'] = next(cyc)
        nodes.loc[nodes['index']==v, 'start_window'] += 24*nodes.loc[nodes['index']==v, 'Dia']
        nodes.loc[nodes['index']==v, 'end_window'] += 24*nodes.loc[nodes['index']==v, 'Dia']
    print(nodes.loc[nodes['index'].isin(mod_indexes)])
    
start_windows = nodes['start_window'].values.astype(np.float)
end_windows = nodes['end_window'].values.astype(np.float)


if nodes['M3'].max() > fleet_df['Capacidades'].max():
    print("Clients with volume bigger than largest vehicle: ",nodes.loc[nodes['M3']-fleet_df['Capacidades'].max()>0, 'index'].values)
    # temporary hack: set biggest vehicle capacity to the max of a given client
    fleet_df.loc[fleet_df['Capacidades']==fleet_df['Capacidades'].max(), 'Capacidades'] = math.ceil(nodes['M3'].max())

'''
Instância inviável:

se u_0 + t_0i > 8 e a janela de tempo do cliente i é no dia 0 (primeiro dia), não tem como servir esse cliente.

se u_0 + t_0i + u_i > 8 e a janela de tempo do cliente é no dia 0, também não é factível 

as duas podem ser representadas pela segunda condição

se volume_i >= max volume_k
'''