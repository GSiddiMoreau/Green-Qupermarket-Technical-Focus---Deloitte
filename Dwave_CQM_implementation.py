# Vineet Mukim for team Qrious


import numpy as np
import pandas as pd
from dimod import Binary, ConstrainedQuadraticModel, cqm_to_bqm
from matplotlib import pyplot as plt
import neal
from dwave.system import LeapHybridCQMSampler, DWaveSampler
from dwave.system.composites import EmbeddingComposite

def read_data_from_excel_file(day, delta_T, sim_car, E_rate_list):
    """
    this function reads the given excel file and populates the data in required matrix form.
    day is first 3 letters of the day, fixed sim_time of 7 AM to 10 PM, deltaT in minutes and
    number of simulated vehicles.
    """
    sim_time = [7, 22]
    T = int((sim_time[1]-sim_time[0])*60/delta_T)
    print('T = ' +str(T))
    day_dict = {'Mon':1, 'Tue':2, 'Wed':3, 'Thu':4, 'Fri':5, 'Sat':6}
    #------------------------------------------------------------------
    # creating weather vector
    weather_data = pd.read_excel('WomaniumChallenge.xlsx', 'Weather Data')
    #print(weather_data.columns)
    T_start = sim_time[0]*2
    T_end = sim_time[1]*2
    #print([T_start, T_end])
    day_weather_data = np.array(weather_data[T_start:T_end] ['Unnamed: '+str(2*day_dict[day]-1)])
    #print(day_weather_data)
    E_sol_dir = np.zeros(T)#shape=(T, 1))
    multiple = int(30/delta_T)
    for i in range(day_weather_data.size):
        if day_weather_data[i]=='Night':
            solar = 0
        elif day_weather_data[i]=='Sun':
            solar = 180
        elif day_weather_data[i]=='Cloudy':
            solar = 90 # changed from 80 to ensure multiple of 30 to reduce # of char/disc rates
        elif day_weather_data[i]=='Rain':
            solar = 30 # changed from 50 to ensure multiple of 30 to reduce # of char/disc rates
        #print([i, solar])
        E_sol_dir[multiple*i:multiple*i+multiple] = solar
    print('E_solar_dir = ' + str(E_sol_dir.transpose()))  
    #------------------------------------------------------------------
    # creating mall requirement vector
    E_mall_req = np.ones(T)*120
    E_mall_req[0:2*multiple] = 30  # changed from 24 to ensure multiple of 30 to reduce # of char/disc rates24
    print('E_mall_req = ' + str(E_mall_req))
    #------------------------------------------------------------------
    # creating car data matrix
    all_car_data = pd.read_excel('WomaniumChallenge.xlsx', 'Car Data')
    #print(all_car_data.dtypes)
    car_dict = {'Mon':127, 'Tue':115, 'Wed':121, 'Thu':111, 'Fri':110, 'Sat':204} # number of cars on that day
    np.random.seed(0)
    selected_cars = np.sort(np.random.choice(range(1, car_dict[day]+1), sim_car, replace=False))
    print('selected_cars = ' + str(selected_cars))
    car_data = np.ndarray(shape=(sim_car, 7))
    for j in range(selected_cars.size):
        a = (all_car_data.loc[all_car_data['Car'] == day[0:2]+str(selected_cars[j]), 'Arrival Time']).values[0]
        a = a.hour + a.minute/60.0
        car_data[j][0] = a
        b = (all_car_data.loc[all_car_data['Car'] == day[0:2]+str(selected_cars[j]), 'Depature Time (End of Slot)']).values[0]
        b = b.hour + b.minute/60.0
        car_data[j][1] = b
        car_data[j][2] = all_car_data.loc[all_car_data['Car'] == day[0:2]+str(selected_cars[j]), 'Arrival Battery in %']
        car_data[j][3] = all_car_data.loc[all_car_data['Car'] == day[0:2]+str(selected_cars[j]), 'Minimal Depature Battery in %']
        car_data[j][4] = int((a-7)*60/delta_T) #starting time slots from 7AM ToDo indexing from 0?
        car_data[j][5] = int((b-7+0.5)*60/delta_T)-1 #ToDo indexing from 0?
        car_data[j][6] = ((b-a)+0.5)*60/delta_T
    print('car_data = ' + str(car_data))
    total_mall_variables = int(2*15*60/delta_T)*len(E_rate_list)
    print('total mall variables (char and disc with all diff char/disc rates) required = ' + str(total_mall_variables))
    total_car_variables = int(2*np.sum(car_data, axis=0)[6])*len(E_rate_list)
    print('total car variables (char and disc with all diff char/disc rates) required = ' + str(total_car_variables))
    total_variables = total_mall_variables + total_car_variables
    print('total variables required = ' + str(total_variables))
    total_constraints = int(total_variables/2/len(E_rate_list) + 2*T + 2*total_car_variables/2/len(E_rate_list) + T)
    # above total includes constrints from semi positive definiteness of E_plug
    # this also ignores trivial <= 0 constraints (simulatenous char/disc) on linear combinations of binary variables
    print('total constraints required = ' + str(total_constraints))
    return T, E_sol_dir, E_mall_req, car_data, total_mall_variables, total_car_variables


def build_CQM(delta_T, T, E_rate_list, total_mall_variables, total_car_variables, E_mall_req, E_sol_dir, car_data):
    """
    this function generates objective (cost) function and constraints for CQM quantum hybrid solver
    """
    total_variables = total_mall_variables + total_car_variables
    c_plug = 420 # emission for plug
    c_solar = 0 # emission for solar
    c_mall = c_solar
    c_car = 84
    #------------------------------------------------------------------
    # objetive function  
    cqm = ConstrainedQuadraticModel()
    tau_list = [Binary(f'tau_{i}') for i in range(total_variables)]
    #print(len(tau_list))
    f = 0 # initialize the objective function
    # adding constant and mall battery terms in objective function
    # the variables are arranged in tau_list as [discharge and charge] for all times for mall battery
    # followed by the same sequence for cars
    # this step populates variables from 0 till 2*mall variables for discharging and charging 
    for i in range(T):
        f = f + c_plug*E_mall_req[i] + (c_solar-c_plug)*E_sol_dir[i]
        for k in range(len(E_rate_list)):
            f = f + (c_mall-c_plug)*E_rate_list[k]*(tau_list[(len(E_rate_list)*i+k)*2]\
                - tau_list[(len(E_rate_list)*i+k)*2+1])
    #print(f)
    # now, we assign car battery variables to tau_list and add to objective function
    increment1 = 0
    for i in range(len(car_data)):
        increment2 = 0
        for j in range(int(car_data[i][4]), int(car_data[i][5])+1):
            for k in range(len(E_rate_list)): 
                f = f + (c_car-c_plug)*E_rate_list[k]*(tau_list[total_mall_variables+increment1+increment2]\
                    - tau_list[total_mall_variables+increment1+increment2+1])
                increment2 = increment2 + 2     
        increment1 = increment1 + increment2
    f = f*delta_T/60.0 #divide by 60.0 to have CO2 emission in g/kWh unit
    print('-----')
    print('total CQM objective (cost) function variables created = ' + str(len(f)))
    #print('CQM objective (cost) function = ' + str(f))
    cqm.set_objective(f)
    #------------------------------------------------------------------
    # constraints
    # battery cant charge and discharge simultaneously at a given time and for a given rate
    # possible options for all batteries are charge, do nothing or discharge
    count = 0
    for i in range(0, total_variables, 2*len(E_rate_list)):
        #print(i)
        #cqm.add_constraint(tau_list[i]+tau_list[i+1] >= 0) # label, removed as it is trivial constraint for binary variables
        d = 0
        for j in range(2*len(E_rate_list)):
            d = d + tau_list[i + j]
        #print(d)
        cqm.add_constraint(d <= 1, label='not allowing simulatenous char/disc at any rate of any battery at binary var sum <=1' + str(i))
        count = count + 1
    #print(tau_list[i]+tau_list[i+1])
    print('non-simultaneous char/disc (for each time and rate) constrained created = ' + str(count))
    # constraints on mall battery capacity to be between >=0 and <=500 kWh at all times except last
    # at 10 PM slot end, the battery must be enough to supply till 7 AM i.e 24 x 9 = 216 kWh
    count = 0
    for i in range(T):
        #print(i)
        j = 0
        g = 0
        while j<=i:
            #print(j)
            increment3 = 0            
            for k in range(len(E_rate_list)):
                g = g + (-tau_list[2*len(E_rate_list)*j+increment3] +\
                    tau_list[2*len(E_rate_list)*j+increment3+1])*E_rate_list[k]
                increment3 = increment3 + 2   
            j = j + 1
        g = g * delta_T/60.0
        #print(g)
        if i==T-1:    
            cqm.add_constraint(g >= 216, label = 'min capacity of mall battery at 10 PM' + str(i)) # enough capacity for 9 hours
        else:
            cqm.add_constraint(g >= 0, label = 'min capacity of mall battery at all other times >=0' + str(i))
        cqm.add_constraint(g <= 500, label = 'max capacity of mall battery at all times <=500' + str(i))
        count = count + 2
    print('min and max mall battery constraints created = ' + str(count))
    #------------------------------------------------------------------
    # constraints on car battery capacities to be between >=0 and <=120 kWh at all times except
    # departure (last) time slot, in which minimum departure battery must be ensured
    count = 0
    increment5 = 0
    for i in range(len(car_data)):
        #print(i)
        i_start = int(car_data[i][4])
        i_end = int(car_data[i][5])
        arrival_charge = int(car_data[i][2]*120/100.0)
        min_departure_charge = int(car_data[i][3]*120/100.0)
        #print(i_start, i_end, arrival_charge, min_departure_charge)
        h = arrival_charge
        for j in range(i_start, i_end+1):
            #print(j)
            k = i_start
            while k<=j:
                #print(k)
                increment4 = 0
                for l in range(len(E_rate_list)):
                    h = h + (-tau_list[total_mall_variables+2*len(E_rate_list)*(k-i_start)+increment4+increment5] + \
                        tau_list[total_mall_variables+2*len(E_rate_list)*(k-i_start)+increment4+increment5+1])*E_rate_list[l]*delta_T/60.0
                    increment4 = increment4 + 2
                k = k + 1
            #print(h)
            if i==i_end:    
                cqm.add_constraint(h >= min_departure_charge, label = 'min departure battery for car' + str([i, j])) # enough capacity for 9 hours
            else:
                cqm.add_constraint(h >= 0, label = 'min level of car battery capacity must be >=0 at all times' + str([i,j]))
            cqm.add_constraint(h <= 120, label = 'min level of car battery capacity must be <=120 at all times' + str([i,j]))
            count = count + 2
        increment5 = increment5 + 2*len(E_rate_list)*(i_end - i_start + 1)
    print('min and max car battery constraints created = ' + str(count))
    #------------------------------------------------------------------
    # constraints of semi positive definiteness of E_plug value, need one constraint eq per time slot
    # this is to avoid solver choosing 420*-ve as path of highest gradient to minimize the emission
    count = 0
    for i in range(T):
        #print(i)
        u = 0
        u = u + E_mall_req[i] - E_sol_dir[i]
        increment6 = 0
        for j in range(len(E_rate_list)):
            #print(j)
            u = u - 1*(tau_list[2*len(E_rate_list)*i+increment6] -\
                tau_list[2*len(E_rate_list)*i+increment6+1])*E_rate_list[j]
            increment6 = increment6 + 2
        #print(u)
        increment7 = 0
        for k in range(len(car_data)):
            k_start = int(car_data[k][4])
            k_end = int(car_data[k][5])
            increment6 = 0
            for l in range(k_start, k_end+1):
                if l == i:
                    for m in range(len(E_rate_list)):
                        #print(m)
                        u = u - 1*(tau_list[total_mall_variables+2*len(E_rate_list)*(l-k_start)+increment6+increment7] -\
                        tau_list[total_mall_variables+2*len(E_rate_list)*(l-k_start)+increment6+increment7+1])*E_rate_list[m]
                        increment6 = increment6 + 2   
            increment7 = increment7 + (k_end - k_start + 1)*2*len(E_rate_list)
        #print(u)
        cqm.add_constraint(u >= 0, label = 'E_plug value must be semi positive definite' + str(i))
        count = count + 1
    print('E_plug semi positive definite constraints created = ' + str(count))
    return cqm


def visualize_results(delta_T, T, E_rate_list, E_sol_dir, E_mall_req, car_data, variable_list):
    time_list = np.arange(7, 22, delta_T/60.0)
    total_mall_variables = T*2*len(E_rate_list)
    time_list = time_list + delta_T/60.0/2 # to shift time axis for better visualization
    #print(time_list)
    E_sol_dir = E_sol_dir*delta_T/60.0 # converting to kWh
    E_mall_bat_disc = np.zeros(len(time_list))
    E_mall_bat_char = np.zeros(len(time_list))
    E_car_bat_disc = np.zeros(len(time_list))
    E_car_bat_char = np.zeros(len(time_list))
    for i in range(T):
        #print('time slot', i)
        for j in range(len(E_rate_list)):
            E_mall_bat_disc[i] = E_mall_bat_disc[i] + variable_list[2*(i*len(E_rate_list)+j)]*E_rate_list[j]*delta_T/60.0
            E_mall_bat_char[i] = E_mall_bat_char[i] + variable_list[2*(i*len(E_rate_list)+j)+1]*E_rate_list[j]*delta_T/60.0
            #print(2*(i*len(E_rate_list)+j), 2*(i*len(E_rate_list)+j)+1, E_rate_list[j])
        #print('---')
        increment8 = 0
        for k in range(len(car_data)):
            #print('car', k)
            k_start = int(car_data[k][4])
            k_end = int(car_data[k][5])
            increment9 = 0
            for l in range(k_start, k_end+1):
                #print('car time slot', l)
                if l == i:
                    #print('car', k)
                    #print('car time slot', l)
                    #print('time slot matches i==l')
                    for m in range(len(E_rate_list)):
                        #print('rate', m)
                        E_car_bat_disc[i] = E_car_bat_disc[i] + variable_list[total_mall_variables+\
                            2*len(E_rate_list)*(l-k_start)+increment8+increment9]*E_rate_list[m]*delta_T/60.0
                        E_car_bat_char[i] = E_car_bat_char[i] + variable_list[total_mall_variables+\
                            2*len(E_rate_list)*(l-k_start)+increment8+increment9+1]*E_rate_list[m]*delta_T/60.0
                        #print(total_mall_variables+2*len(E_rate_list)*(l-k_start)+increment8+increment9)
                        #print(total_mall_variables+2*len(E_rate_list)*(l-k_start)+increment8+increment9+1)  
                        #print('-next time slot-')
                        increment9 = increment9 + 2
            increment8 = increment8 + (k_end - k_start + 1)*2*len(E_rate_list)
            #print('-next car-')
    # calculate E_plug
    E_plug = (E_mall_req - E_sol_dir - E_mall_bat_disc + E_mall_bat_char - E_car_bat_disc + E_car_bat_char)
    #print(E_plug)
    #------------------------------------------------------------------
    # plotting
    width = 0.45
    plt.bar(time_list, E_sol_dir, width=width, label='solar', color='gold')
    plt.bar(time_list, E_plug, width=width, label='plug', color='cyan', bottom = E_sol_dir+E_mall_bat_disc+E_car_bat_disc)
    plt.bar(time_list, E_mall_bat_disc, width=width, label='market battery discharge', color='silver', bottom = E_sol_dir)
    plt.bar(time_list, -1*E_mall_bat_char, width=width, label='market battery charge', color='pink')    
    plt.bar(time_list, E_car_bat_disc, width=width, label='car battery discharge', color='olive', bottom = E_sol_dir+E_mall_bat_disc)   
    plt.bar(time_list, -1*E_car_bat_char, width=width, label='car battery charge', color='black', bottom = -1*E_mall_bat_char)
    plt.ylabel("Energy Generation (+ve) and Consumption (-ve) (kWh)", fontsize = 8)
    plt.xlabel("Time (24h Format)", fontsize = 8)
    plt.xticks(fontsize = 8)
    plt.yticks(fontsize = 8)
    plt.legend(loc = 'lower center', fontsize = 8, ncol = 3)
    plt.suptitle("Qupermarket Energy Options Scheduling and Distribution", fontsize = 10)
    plt.title('for $CO_2$ Emission Minimization', fontsize = 10)
    plt.show()


def main():
    # note: some numbers like mall consumption at night and solar generation are modified to be multiple of 30 kW
    # this reduces required entries (degrees of freedom) in E_req_list :)
    day = 'Mon'
    delta_T = 30 # must be <= 30, with diff char/disc rates, we dont need very low deltaT
    sim_car = 10 #127
    E_rate_list = [30, 60, 90, 120] # discretized charging and discharging rates
    T, E_sol_dir, E_mall_req, car_data, total_mall_variables, total_car_variables\
        = read_data_from_excel_file(day, delta_T, sim_car, E_rate_list)
    cqm = build_CQM(delta_T, T, E_rate_list, total_mall_variables, total_car_variables,\
        E_mall_req, E_sol_dir, car_data)
    

    # classical solver
    bqm, invert = cqm_to_bqm(cqm, 10)
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads = 100)
    #print(sampleset.first.sample)
    #print(len(sampleset.first.sample.values()))
    #print(sampleset.first.sample.keys())
    #print(sampleset.record)
    #print(sampleset.first.energy)
    #print(len(sampleset.record))
    '''
    # hybrid CQM quantum solver  
    sampler = LeapHybridCQMSampler()
    sampleset = sampler.sample_cqm(cqm)
    #print(sampleset.first.sample)
    #print(sampleset.info)
      
    # hybrid BQM quantum solver
    bqm, invert = cqm_to_bqm(cqm, 10000)
    sampler = DWaveSampler(solver={'topology__type': 'pegasus'})
    sampler = EmbeddingComposite(sampler)
    sampleset = sampler.sample(bqm, label='Example')
    '''
    # visualizing results
    variable_list = np.empty(total_mall_variables + total_car_variables)
    for i in range(total_mall_variables + total_car_variables):
        variable_list[i] = sampleset.first.sample['tau_'+str(i)]
    #print(variable_list)
    #variable_list = np.random.choice(2, int(total_mall_variables+total_car_variables), replace=True)
    visualize_results(delta_T, T, E_rate_list, E_sol_dir, E_mall_req, car_data, variable_list)

if __name__=='__main__':
    main()