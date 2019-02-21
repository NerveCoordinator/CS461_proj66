import random
import simpy
import math

RANDOM_SEED = 42
NUM_DOCTORS = 2 
HEALTIME = 4
T_INTER = 2 
SIM_TIME = 720 #Goal is 12 hour window
ARRIVAL_RATE = 0.4

#WITHIOUT THE USE OF PREDICTOR DATA, HERE IS THE HOSPITAL METRICS WE WILL USE:
#http://emergencias.portalsemes.org/descargar/evidence-of-the-validity-of-the-emergency-severity-index-for-triage-in-a-general-hospital-emergency-department/force_download/
#USING THE ESI METRIC, OUR GLOBAL CHANCE OF BEING EACH VALUE WILL BE THE FOLLOWING
ESI1 = 0.7
ESI2 = 14.9
ESI3 = 36.6
ESI4 = 35.1
ESI5 = 12.7
#USING THE RESOURCE METRICS, WE GIVE THE FOLLOWING CHANCES FOR CONSUMING RESOURCES
ESI1_CONSUME_0 = 73.5
ESI1_CONSUME_1 = 16.3
ESI1_CONSUME_M = 10.2
ESI2_CONSUME_0 = 8.6
ESI2_CONSUME_1 = 82.1
ESI2_CONSUME_M = 9.3
ESI3_CONSUME_0 = 3.4
ESI3_CONSUME_1 = 11.6
ESI3_CONSUME_M = 85
ESI4_CONSUME_0 = 6.6
ESI4_CONSUME_1 = 8.2
ESI4_CONSUME_M = 82.2
ESI5_CONSUME_0 = 0
ESI5_CONSUME_1 = 0
ESI5_CONSUME_M = 100
#USING THE TIME METRICS, WE GIVE THE FOLLOWING TIME WITH STANDARD DEVIATION FOR THEIR STAY
#THESE TIME VALUES ARE IN MINUETS
ESI1_AVG_TIME = 476
ESI1_TIME_SD  = 228
ESI2_AVG_TIME = 716
ESI2_TIME_SD  = 659
ESI3_AVG_TIME = 333
ESI3_TIME_SD  = 259
ESI4_AVG_TIME = 176
ESI4_TIME_SD  = 110
ESI5_AVG_TIME = 166
ESI5_TIME_SD  = 93

ESI_CHANCE   = [ESI1, ESI2, ESI3, ESI4, ESI5]
ESI_CONSUME  = [[ESI1_CONSUME_0, ESI1_CONSUME_1, ESI1_CONSUME_M],
				[ESI2_CONSUME_0, ESI2_CONSUME_1, ESI2_CONSUME_M],
				[ESI3_CONSUME_0, ESI3_CONSUME_1, ESI3_CONSUME_M],
				[ESI4_CONSUME_0, ESI4_CONSUME_1, ESI4_CONSUME_M],
				[ESI5_CONSUME_0, ESI5_CONSUME_1, ESI5_CONSUME_M]]
ESI_TIME	 = [[ESI1_AVG_TIME, ESI1_TIME_SD],
				[ESI2_AVG_TIME, ESI2_TIME_SD],
				[ESI3_AVG_TIME, ESI3_TIME_SD],
				[ESI4_AVG_TIME, ESI4_TIME_SD],
				[ESI5_AVG_TIME, ESI5_TIME_SD]]



import numpy as np
from numpy.random import RandomState
RNG_SEED = 800
prng = RandomState(RNG_SEED)

class Record(object):
	def __init__(self):
		self.doctors = 0
		self.beds = 0
		self.patients = 0
		self.curr_waits = []
		self.history = []

	def new_history(self, new_docs, new_beds):
		if (len(self.curr_waits) > 0):
			self.history.append((self.patients, len(self.curr_waits), self.doctors, self.beds, self.curr_waits))
			self.curr_waits = []
			self.doctors = new_docs
			self.beds = new_beds
			self.patients = 0

	def new_patient(self):
		self.patients += 1

	def new_wait(self, wait):
		self.curr_waits.append(wait)




class Hospital(object):
	#HERE WE WILL DEFINE THE ELEMENTS THAT THE HOSPITAL WILL HAVE
	def __init__(self, env, num_doctors, num_beds):
		self.env = env
		self.doctor = num_doctors
		self.beds = num_beds
		self.patients = [] #HERE WE NEED TO IMPLEMENT THE CARRY OVER PATIENTS BETWEEN HOSPITAL DAYS
		self.bed_contents = []
		self.discharged = 0

	def recieve_patient(self, env, patient):
		cur_patient_esi = patient.status
		i = 0
		arr_len = len(self.patients)
		while(i < arr_len):
			if(cur_patient_esi < self.patients[i].status):
				self.patients.insert(i, patient)
				i = arr_len + 1
			else:
				i += 1
		if(i == arr_len):
			self.patients.append(patient)
		print("Waiting	Patient    " + str(patient.id) + " Status = " + str(patient.status) + 
			" at time: " + str(env.now)) 


	def check_on_patients(self):
		arr_len = len(self.bed_contents)
		if(arr_len > 0):
			i = 0
			d = 0 #Number of doctors in use			
			while(i < arr_len and d < self.doctor):
				if(self.bed_contents[i].time_with_doc > 0):
					d += 1
					self.bed_contents[i].time_with_doc -= 1
				i += 1
	
	#Will update patient's time and remove "cured" patients
	def update_patient(self, env):
		arr_len = len(self.bed_contents)
		if(arr_len > 0):
			i = 0
			while(i < arr_len):
				if(self.bed_contents[i].time_with_doc <= 0 and self.bed_contents[i].time_to_heal <= 0):
					print("Discharged Patient " + str(self.bed_contents[i].id) + " Status = " + str(self.bed_contents[i].status) + 
						" at time: " + str(env.now))
					self.discharged += 1					
					self.bed_contents.pop(i) #can be recorded if we want
					arr_len -= 1
				else:
					self.bed_contents[i].time_to_heal -= 1 #could overflow if patient waits for eternity 
					i += 1
	
	#Will add new patients to beds if they are avalible
	def add_to_beds(self, env):
		arr_len = len(self.bed_contents)
		j = 0
		while(arr_len < self.beds and len(self.patients) > 0):
			cur_patient_esi = self.patients[0].status
			i = 0
			#find where in bed order the patient belongs and place them there
			#Will insert patient in ESI order with 1 being front, 5 beint in back
			while(i < arr_len):
				if(self.bed_contents[i].status < cur_patient_esi): 
					#Found spot to insert new patient into bed
					self.bed_contents.insert(i, self.patients.pop(0))
					j = i
					arr_len += 1
					i = arr_len + 1
				else:
					i += 1

			#lowest classified case at the moment
			if(i == arr_len):
				self.bed_contents.append(self.patients.pop(0))
				j = i
				arr_len += 1
			print("Admitted   Patient " + str(self.bed_contents[j].id) + " Status = " + str(self.bed_contents[j].status) + 
				" at time: " + str(env.now))

#IMPORTANT#
#Currently there is a proplem with how hospital patients are catagorized, currenlty patient 5's will wait
#Forever as other patients of equal status are serviced

	def pass_time(self, env):
		#pass_time will be the general function that can be used to sequence a minute
		self.check_on_patients()
		self.update_patient(env)
		self.add_to_beds(env)



class patient(object):
	def __init__(self, env, status, consume, time_to_heal, arrival_time, id):
		self.env = env
		self.status = status
		self.consume = consume
		self.time_to_heal = time_to_heal
		self.time_with_doc = time_to_heal/4 #Using the source http://ugeskriftet.dk/files/scientific_article_files/2018-12/a4558.pdf
		self.arrival_time = arrival_time
		self.id = id



class patient_generator(object):
	def __init__(self, env):
		self.env = env
		self.esi_chance	    = ESI_CHANCE
		self.esi_consume	= ESI_CONSUME
		self.esi_time	    = ESI_TIME
		self.total_patients = 0



	def make_patient(self, env):
		pat_esi = self.get_status(env)
		pat_com = self.get_consume(env, pat_esi)
		pat_tim = self.get_time(env, pat_esi)
		new_pat = patient(env, pat_esi, pat_com, pat_tim, env.now, self.total_patients)
		self.total_patients += 1
		return (new_pat)

	#The esi status of the patient upon them entering the hospital
	def get_status(self, env):
		i = 5
		while(i > 1):
			if(self.esi_chance[i - 1] < random.uniform(0,100)):
			#if(ESI_CHANCE[int(i - 1)] < random.uniform(0, 100)):
				return i
			i -= 1
		return 1

	#The amount of resources that the patient will consume during their visit
	def get_consume(self, env, esi):
		i = 3
		while(i > 1):
			if(self.esi_consume[esi - 1][i - 1] < random.uniform(0,100)):
				return i
			i -= 1
		return (self.esi_consume[esi - 1][0])

	#How much time that the patient will need in the bed in order to be discharged from the ER
	def get_time(self, env, esi):
		return (self.esi_time[esi - 1][0] + self.esi_time[esi - 1][1] * random.uniform(.5, 1.5))


def setup(env, num_doctors, num_beds, previous_patients):

	hospital = Hospital(env, num_doctors, num_beds)
	mafia = patient_generator(env) #JOKE

	#This is where we will run the simulation loop
	#I'm writing this on 2 assumptions:
	#Each iteration of the while loop is one minute
	#Corvallis has the odds of 0.0477495 each minute of a patient showing up to the emergency department.
	i = 0
	while True:
		patient_chance = random.random()
		if(patient_chance < 0.0477495):
			hospital.recieve_patient(env, mafia.make_patient(env))
		hospital.pass_time(env)
		if(env.now % 100 == 0):
			print("\nHOSPITAL CURRENT STATS AT TIME: " + str(env.now))
			print("PATIENTS WAITING: "  + str(len(hospital.patients)))
			print("PATIENTS HEALED:  " + str(hospital.discharged) + "\n")
		yield env.timeout(1)


def simulate(num_doctors, num_beds, sim_time, previous_patients):
	print('New Hospital')
	random.seed(RANDOM_SEED)  

	env = simpy.Environment()
	env.process(setup(env, num_doctors, num_beds, previous_patients))
	
	print("\n*")
	print("Starting  Doctors: " + str(num_doctors) + " Beds: " + str(num_beds))
	env.run(until=sim_time)
	print("\nFinishing Doctors: " + str(num_doctors) + " Beds: " + str(num_beds))
	print("*\n")


'''
HERE IS WHERE OUR PROGRAM IS ACTUALLY BEING RUN
Order of functions:

Record Object is initialized:
	Maintains record of hospitalized paitents and the time to help

Simulate Object:
	Simulate sets up simpy enviroment to equal the setup function
	Also takes in basic inputs for the simulation
	Have the process run until time is "up"

Setup:
	Takes basic inputs from simulate
	Initializes the Hospital 
	Randomly Generate patients for hospital forever
'''

print("Starting the simulator!")
for i in range(8,12,1): #beds
	for j in range(3,5,1): #doctors
		simulate(j, i, SIM_TIME, [])

