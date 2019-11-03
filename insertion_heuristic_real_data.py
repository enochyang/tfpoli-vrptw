'''
Heurística de inserção para o problema de roteirização de veículos

Características: janelas de tempo, frota heterogênea, descanso obrigatório
TODO: refactor, separar as funções em diferentes arquivos
'''

from operator import itemgetter
import math

import functools
import time
import pandas as pd


def timer(func):
    '''Print the execution time of the decorated function'''
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value, run_time
    return wrapper_timer


def get_vehicle(clients, verbose=False):
    '''get the vehicle with lowest fixed cost that can serve the clients'''
    total_client_volume = sum([demand_volume[c] for c in clients])
    total_client_value = sum([demand_value[c] for c in clients])
    early_start_time = compute_early_start_time(clients)
    late_start_time = compute_late_start_time(clients)
    for k in sorted(fleet_list, key=lambda x: fixedCost_list[x]):
        if (not vehicle_in_use[k] and #QWeight[k] >= demand_weight and
                QVolume_list[k] >= total_client_volume and QValue_list[k] >= total_client_value
                #and vehicle_start_hour[k]<=early_start_time
                #and vehicle_start_hour[k]>=late_start_time
           ):
            #print("Not in use, volume and value ok")
            accepted = [vehicle_acceptance[c][k] for c in clients]
            #print("is it accepted? ", all(accepted))
            if not all(accepted):
                continue
            try:
                get_service_times_(clients, k, verbose)
                return k
            except:
                continue
    if verbose:
        print("ERROR: No vehicle available to serve these clients: "+str(clients))
        #print("Total weight: "+str(demand_weight))
        print("Total volume: "+str(total_client_volume))
        print("Total value: "+str(total_client_value))
    assert False, "No vehicles"
    return
    #raise Exception("no_vehicles")

def get_largest_vehicle_available(clients=[], verbose=False):
    '''get the largest vehicle that can serve the clients'''
    candidates = []
    for k in fleet_list:
        if not vehicle_in_use[k]:
            accepted = [vehicle_acceptance[c][k] for c in clients]
            if all(accepted):
                candidates.append(k)
    if not candidates:
        if verbose:
            print("largest vehicle function failed, no vehicles are available")
        return None
    #weight_capacities = [QWeightByType[fleetMap[k]] for k in candidates]
    volume_capacities = [QVolume_list[k] for k in candidates]
    value_capacities = [QValue_list[k] for k in candidates]
    # assuming that the limiting factor is the volume
    return candidates[volume_capacities.index(max(volume_capacities))]


def get_service_times_(clients, vehicle, verbose=False):
    #print(vehicle)
    sh = vehicle_start_hour[vehicle]
    #print(sh)
    svc_times = [sh for _ in clients] # all start at time of vehicle
    for i, c in enumerate(clients):
        #print(svc_times)
        if c == 0:
            continue
        sjk = (svc_times[i-1] + service_time[clients[i-1]]
               + get_time(clients[i-1], c, vehicle))
        if ((svc_times[i-1] + service_time[clients[i-1]] - sh)%day
            + get_time(clients[i-1], c, vehicle) > work_shift):
            sjk += (rest_shift * 
                    int(((svc_times[i-1] + service_time[clients[i-1]] - sh)%day
                        + get_time(clients[i-1], c, vehicle)) / work_shift))
        #if (svc_times[i-1] + service_time[clients[i-1]] - sh)%day + get_time(clients[i-1], c, self.vehicle) + service_time[c] > work_shift:
        #    sjk += rest_shift*int(((svc_times[i-1] + service_time[clients[i-1]] - sh)%day + get_time(clients[i-1], c, self.vehicle)) / work_shift)
        if sjk > end_windows[c]:
            if verbose:
                print("This route is infeasible, time window violation on client "
                      +str(c)+" in position "+str(i))
            assert False
        svc_times[i] = max(start_windows[c], sjk)
        #print(svc_times)
        if (svc_times[i] + service_time[c] - sh)%day > work_shift:
            if verbose:
                print("This route is infeasible, work shift violation on client "
                      +str(c)+" in position "+str(i))
            assert False
    return svc_times



class Route:
    '''Route object that holds clients, a vehicle and can return some parameters of the route'''
    def __init__(self, client, fake=False):
        self.clients = [0, client, n+1]
        self.vehicle = get_vehicle(self.clients)
        if not fake:
            vehicle_in_use[self.vehicle] = True
        self.update_route_capacity()
        self.service_times = self.get_service_times(self.clients)

    def __repr__(self):
        route_str = "Clients in route: "+str(self.clients)+"\n"
        route_str += "Vehicle: "+str(self.vehicle)+"\n"
        route_str += "Vehicle type: "+str(fleet_df['Veiculos'].iloc[self.vehicle])+"\n"
        #route_str += "Weight used: "+str(self.total_weight)+"/"+str(self.weight_capacity)+"\n"
        route_str += "Volume used: "+str(self.total_volume)+"/"+str(self.volume_capacity)+"\n"
        route_str += "Value used: "+str(self.total_value)+"/"+str(self.value_capacity)+"\n"
        route_str += "Total distance: "+str(self.get_total_distance())+"\n"
        route_str += "Service times: "+str(self.get_service_times(self.clients))
        return route_str

    def update_route_capacity(self):
        #self.weight_capacity = QWeight[self.vehicle]
        self.volume_capacity = QVolume_list[self.vehicle]
        self.value_capacity = QValue_list[self.vehicle]
        #self.total_weight = 0
        self.total_volume = 0
        self.total_value = 0
        for c in self.clients:
            #self.total_weight += sum([d * m for d, m in zip(demand[c], weight)])
            self.total_volume += demand_volume[c]
            self.total_value += demand_value[c]
        #assert (self.weight_capacity>=self.total_weight),"Weight capacity violation"
        assert (self.volume_capacity >= self.total_volume), "Volume capacity violation"
        assert (self.value_capacity >= self.total_value), "Value capacity violation"

    def check_capacity(self, client):
        # check if client fits in route
        return (#(self.weight_capacity >= sum([d * m for d, m in zip(demand[client], weight)])) and
            (self.volume_capacity >= demand_volume[client]) and
            (self.value_capacity >= demand_value[client]))

    def insert_client(self, client, i, verbose=False):
        '''Insert client in position i'''
        assert self.check_capacity(client), "Capacity violation"
        assert i > 0, "Cannot insert in position 0"
        assert i < n+1, "Cannot insert in last position"
        #assert dist_dict.get((self.clients[i-1], client, self.vehicle)) is not None, "No route between previous client and this one for the current vehicle"
        #assert dist_dict.get((client, self.clients[i], self.vehicle)) is not None, "No route between previous client and this one for the current vehicle"
        assert get_dist(self.clients[i-1], client, self.vehicle) is not None, "No route between previous client and this one for the current vehicle"
        assert get_dist(client, self.clients[i], self.vehicle) is not None, "No route between previous client and this one for the current vehicle"
        self.get_service_times(self.clients[0:i]+[client]+self.clients[i:]) # assert feasible
        self.clients.insert(i, client)
        try:
            self.update_route_capacity()
        except:
            vehicle_in_use[self.vehicle] = False
            self.vehicle = get_vehicle(self.clients)
            vehicle_in_use[self.vehicle] = True
            self.update_route_capacity()
        self.service_times = self.get_service_times(self.clients)

    def get_total_distance(self):
        total_dist = 0
        for i in range(len(self.clients)-1):
            total_dist += get_dist(self.clients[i], self.clients[i+1], self.vehicle)
        return total_dist

    def get_service_times_old(self, clients, verbose=False):
        svc_times = [0 for _ in clients] # all start at 0
        for i, c in enumerate(clients):
            if c == 0:
                continue
            else:
                sjk = svc_times[i-1] + service_time[i-1] + get_time(clients[i-1], c, self.vehicle)
                if (svc_times[i-1] + service_time[i-1])%day + get_time(clients[i-1], c, self.vehicle) > work_shift:
                    sjk += rest_shift*int(((svc_times[i-1] + service_time[i-1])%day + get_time(clients[i-1], c, self.vehicle)) / work_shift)
                if sjk > end_windows[c]:
                    if verbose:
                        print("This route is infeasible, time window violation on client "+str(c))
                    assert False
                svc_times[i] = max(start_windows[c], sjk)
        return svc_times

    def get_service_times(self, clients, verbose=False):
        return get_service_times_(clients, self.vehicle, verbose)

    def compute_route_cost(self):
        dists = 0
        for i, c in enumerate(self.clients):
            if c == 0:
                continue
            else:
                dists += get_dist(self.clients[i-1], self.clients[i], self.vehicle)
        #return freightCostByType[fleetMap[self.vehicle]]*dists + fixedCostByType[fleetMap[self.vehicle]]
        return freightCost_list[self.vehicle]*dists + fixedCost_list[self.vehicle]

    def to_dataframe(self):
        self.service_times = self.get_service_times(self.clients)
        records = []
        for i, client in enumerate(self.clients):
            singleton = {'Veículo': fleet_df['Veiculos'].iloc[self.vehicle],
                         'ID Veiculo': self.vehicle,
                         'Sequencia': i,
                         'index': client,
                         'Loja': nodes.loc[nodes['index'] == client, 'Localidade'].iloc[0],
                         'Horario de serviço': self.service_times[i],
                         'Dia Chegada': int(self.service_times[i] // day),
                        }
            records.append(singleton)
        return pd.DataFrame.from_records(records)


#################
##### Costs  ####
#################

def compute_c11(route, client, i, verbose=False):
    '''Increase in distance from adding client in position i+1'''
    return (get_dist(route.clients[i], client, route.vehicle) +
            get_dist(client, route.clients[i+1], route.vehicle) -
            get_dist(route.clients[i], route.clients[i+1], route.vehicle))

def compute_c12(route, client, i, verbose=False):
    # tries to compute the service time increase. If infeasible will assert error, and then we return infinity
    try:
        hyp_svc_times = route.get_service_times(route.clients[0:i+1]+[client]+route.clients[i+1:], verbose)[i+1+1] # we insert new client in i+1, so previous i+1 becomes i+2
    except:
        if verbose:
            print("c12 infinity")
        hyp_svc_times = math.inf
    return hyp_svc_times - route.service_times[i+1]

def compute_c13(route, client, i, verbose=False):
    '''If capacity is not enough, return increase in fixed cost due to change in vehicle'''
    if route.check_capacity(client):
        return 0
    else:
        try:
            new_vehicle = route.get_vehicle(route.clients+[client], verbose)
        except:
            if verbose:
                print("c13 infinity")
            return math.inf
        return (fixedCost_list[new_vehicle] - fixedCost_list[route.vehicle] +
                (freightCost_list[new_vehicle] - freightCost_list[route.vehicle]) * route.get_total_distance())

def compute_c1(route, client, weights, verbose=False):
    min_cost = math.inf
    position = 0
    for i in range(len(route.clients)-1):
        new_cost = (weights[0]*compute_c11(route, client, i, verbose) +
                    weights[1]*compute_c12(route, client, i, verbose) +
                    weights[2]*compute_c13(route, client, i, verbose))
        if new_cost < min_cost:
            min_cost = new_cost
            position = i+1
    return (min_cost, position)

def compute_c2(route, client, weights_c1, weights_c2, verbose=False):
    big_distance = math.inf
    c1, position = compute_c1(route, client, weights_c1, verbose)
    try:
        new_route = Route(client, fake=True)
        #print("Fake route:", new_route)
        #print("vehicleinuse:", vehicle_in_use)
        #vehicle_in_use[new_route.vehicle] = False
        #print("vehicleinuse2:", vehicle_in_use)
        c2 = (weights_c2[0]*new_route.get_total_distance() +
              weights_c2[1]*(get_time(0, client, new_route.vehicle) +
                             new_route.service_times[1]) +
              #weights_c2[2]*fixedCost_list[new_route.vehicle]
              weights_c2[2]*(fixedCost_list[new_route.vehicle] + freightCost_list[new_route.vehicle]*new_route.get_total_distance())
              - c1)
    except:
        if verbose:
            print("Error computing c2 for this client, probably we cannot create a new route")
        if math.isinf(c1):
            if verbose:
                print("c1 is infinite in this case")
            c2 = -math.inf
        else:
            if verbose:
                print("c1 is finite in this case")
            c2 = math.inf
    return client, c2, c1, position


@timer
def insertion_heuristic(weights_c1, weights_c2, verbose=False):
    global vehicle_in_use
    vehicle_in_use = [False for k in fleet_list]
    routes = []
    customer_routed = [True] + [False for c in customers_list] + [True]
    while not all(customer_routed):
        unrouted = [i for i, c in enumerate(customer_routed) if not customer_routed[i]]
        #unrouted_dist = [dist_dict[0, i] for i in unrouted]
        #unrouted_dist = [max([dist_dict[key] for key in dist_dict.keys() if key[0]==0 and key[1]==i])
        #                 for i in unrouted]
        unrouted_dist = [max([get_dist(0, i, k) for k in fleet_list])
                         for i in unrouted]
        if verbose:
            print("Unrouted clients total: ", len(unrouted))
        target_client = unrouted[unrouted_dist.index(max(unrouted_dist))]
        route = Route(target_client)
        unrouted.remove(target_client)
        customer_routed[target_client] = True
        if verbose:
            print("Target client: ", target_client)
            print("Vehicles in use: ", vehicle_in_use)
        largest_vehicle = get_largest_vehicle_available(clients=route.clients, verbose=verbose)
        if not largest_vehicle:
            largest_vehicle = route.vehicle
        QVolume = QVolume_list[largest_vehicle] - route.total_volume
        QValue = QValue_list[largest_vehicle] - route.total_value
        candidates = [c for c in unrouted if
                      (#(QWeight >= sum([d * m for d, m in zip(demand[c], weight)])) and
                         (QVolume >= demand_volume[c]) and
                         (QValue >= demand_value[c]) and
                         vehicle_acceptance[c][route.vehicle])]
        if verbose:
            print("Candidates:", candidates)
        while len(candidates) > 0:
            #print("Vehicles in use pre c2: ", vehicle_in_use)
            C2 = [compute_c2(route, c, weights_c1, weights_c2, verbose) for c in candidates]
            #print("Vehicles in use after c2: ", vehicle_in_use)
            pC2 = [(client, c2, c1, position) for client, c2, c1, position in C2 if c2 > 0]
            if len(pC2) > 0:
                client, _, _, position = max(pC2, key=itemgetter(1))
                if verbose:
                    print("C2:", pC2)
                    print("Selected client: ", client, " at position ",position)
                route.insert_client(client, position, verbose)
                unrouted.remove(client)
                if verbose:
                    print("Routed client: ", client)
                    print("Current route:", route)
                    #print("Remaining unrouted: ", unrouted)
                    print("Vehicles in use: ", vehicle_in_use)
                customer_routed[client] = True
                largest_vehicle = get_largest_vehicle_available(clients=route.clients, verbose=verbose)
                if not largest_vehicle:
                    largest_vehicle = route.vehicle
                QVolume = QVolume_list[largest_vehicle] - route.total_volume
                QValue = QValue_list[largest_vehicle] - route.total_value
                candidates = [c for c in unrouted if
                              (#(QWeight >= sum([d * m for d, m in zip(demand[c], weight)])) and
                                (QVolume >= demand_volume[c]) and
                                (QValue >= demand_value[c]) and
                                vehicle_acceptance[c][route.vehicle])]
                if verbose:
                    print("Candidades: ", candidates)
            else:
                if verbose:
                    print("No candidates for current route, starting new one")
                break
        routes.append(route)
        if verbose:
            print("Latest ROUTE: ", routes[-1])
    return routes


time_const = arcs['Tempo (h)'].max()
dist_const = arcs['Distancia (km)'].max()
cf_const = capacidades['Custo Fixo'].max()
cv_const = capacidades['Frete Km'].max()

def normalize_tables(time_const=time_const, dist_const=dist_const, cf_const=cf_const, cv_const=cv_const):
    global service_time
    global arcs
    global work_shift
    global rest_shift
    global day
    global vehicle_start_hour
    global start_windows
    global end_windows
    global fixedCost_list
    global freightCost_list
    service_time /= time_const
    arcs['Tempo (h)'] /= time_const
    work_shift /= time_const
    rest_shift /= time_const
    day /= time_const
    start_windows /= time_const
    end_windows /= time_const
    vehicle_start_hour /= time_const
    arcs['Distancia (km)'] /= dist_const
    fixedCost_list /= cf_const
    freightCost_list /= cv_const


def denormalize_tables():
    normalize_tables(1/time_const, 1/dist_const, 1/cf_const, 1/cv_const)


def routes_to_df(routes):
    df_list = []
    for i, r in enumerate(routes):
        df_r = r.to_dataframe()
        df_r['ID Viagem'] = i + 1
        df_list.append(df_r)
    return pd.concat(df_list, axis=0)


def compute_cost_route_idx(routes_df, idx, add_cd=False):
    route = routes_df[routes_df['ID Viagem'] == idx]
    vehicle = route['Veículo'].unique()[0]
    cf = capacidades.loc[capacidades['Veiculos'] == vehicle, 'Custo Fixo'].iloc[0]
    clients = list(route.sort_values('Sequencia')['index'].values)
    if add_cd:
        clients = [0] + clients + [n+1]
    dist = sum([get_dist(i, j, vehicle)
                for i, j in pairwise(clients)])
    cv = dist*capacidades.loc[capacidades['Veiculos'] == vehicle, 'Frete Km'].iloc[0]
    return cf + cv


def compute_cost_routes_df(routes_df, add_cd=False):
    return sum([compute_cost_route_idx(routes_df, idx, add_cd) for idx in routes_df['ID Viagem'].unique()])
