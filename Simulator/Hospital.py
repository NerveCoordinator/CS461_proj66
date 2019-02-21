import random
import simpy
import math
import operator
import numpy as np
from numpy.random import choice
from numpy.random import RandomState

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

NUM_DOCTORS = 2 
HEALTIME = 4
T_INTER = 2 
SIM_TIME = 720 #Goal is 12 hour window
ARRIVAL_RATE = 0.4

# Hospital metrics used instead of predictor data:
# http://emergencias.portalsemes.org/descargar/evidence-of-the-validity-of-the-emergency-severity-index-for-triage-in-a-general-hospital-emergency-department/force_download/

# Using ESI metric, global chance of being each value:
ESI1 = 0.7
ESI2 = 14.9
ESI3 = 36.6
ESI4 = 35.1
ESI5 = 12.7

ESI_CHANCE   = [ESI1, ESI2, ESI3, ESI4, ESI5]

# Using resouce metrics, chances for resource consumption:

ESI1_CONSUME_0 = 73.5
ESI1_CONSUME_M = 10.2
ESI1_CONSUME_1 = 16.3


ESI2_CONSUME_0 = 8.6 
ESI2_CONSUME_1 = 82.1
ESI2_CONSUME_M = 9.3

ESI3_CONSUME_0 = 3.4
ESI3_CONSUME_1 = 11.6
ESI3_CONSUME_M = 85


ESI4_CONSUME_0 = 6.6
ESI4_CONSUME_1 = 8.2
ESI4_CONSUME_M = 85.2 #82.2 # original value # original sums to 97! So I added 3 to M

ESI5_CONSUME_0 = 0
ESI5_CONSUME_1 = 0
ESI5_CONSUME_M = 100

ESI_CONSUME  = [[ESI1_CONSUME_0, ESI1_CONSUME_1, ESI1_CONSUME_M],
		[ESI2_CONSUME_0, ESI2_CONSUME_1, ESI2_CONSUME_M],
		[ESI3_CONSUME_0, ESI3_CONSUME_1, ESI3_CONSUME_M],
		[ESI4_CONSUME_0, ESI4_CONSUME_1, ESI4_CONSUME_M],
		[ESI5_CONSUME_0, ESI5_CONSUME_1, ESI5_CONSUME_M]]

# Using time metrics, here are standard deviation of stay time in minutes:

ESI1_AVG_TIME = 476
ESI2_AVG_TIME = 716
ESI3_AVG_TIME = 333
ESI4_AVG_TIME = 176
ESI5_AVG_TIME = 166

ESI1_TIME_SD  = 228
ESI2_TIME_SD  = 659
ESI3_TIME_SD  = 259
ESI4_TIME_SD  = 110
ESI5_TIME_SD  = 93

ESI_TIME     = [[ESI1_AVG_TIME, ESI1_TIME_SD],
		[ESI2_AVG_TIME, ESI2_TIME_SD],
		[ESI3_AVG_TIME, ESI3_TIME_SD],
		[ESI4_AVG_TIME, ESI4_TIME_SD],
		[ESI5_AVG_TIME, ESI5_TIME_SD]]

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
	def __init__(self, env, num_doctors, num_beds):
		self.env          = env
		self.beds         = num_beds
		self.patients     = [] 
		self.bed_contents = []
		self.discharged   = 0
		self.doctors      = num_doctors
		self.available_doctors = num_doctors

	def recieve_patient(self, env, patient):
		self.patients.append(patient) 
		print("Waiting	Patient    " + str(patient.id) + " Status = " + str(patient.status) + " at time: " + str(env.now)) 

	def check_on_patients(self):
		for patient in self.bed_contents:
			if (self.available_doctors > 0):
				self.available_doctors -= 1
				patient.time_with_doc -= 1				
			else:
				return
	
	# Updates patient times and removes "cured" patients
	def update_patient(self, env):
		# keep in beds everyone that isn't discharged (python can't edit lists during for loop)
		new_bed_contents = [] 
		for patient in self.bed_contents:
			if (patient.time_with_doc <= 0 and patient.time_to_heal <= 0):
				print("Discharged Patient " + str(patient.id) + " Status = " + str(patient.status) + " at time: " + str(env.now))
				self.discharged += 1				
			else:
				patient.time_to_heal -= 1
				new_bed_contents.append(patient) 
		self.bed_contents = new_bed_contents

	
	# Will add new patients to beds if they are avalible
	# (Currently sorts everyone rather than individually removing/adding)
	def add_to_beds(self, env):	
		full_patient_list = self.bed_contents	+ self.patients
		full_patient_list.sort(key=operator.attrgetter('status'))
		full_patient_list.sort(key=operator.attrgetter('time_to_heal'))	#so anyone who was using the bed has a chance to keep it. 	
		self.bed_contents = full_patient_list[:self.beds]
		self.patients = full_patient_list[self.beds:]
		
		#no longer displays this, but should. But we also need to patients getting kicked out of beds...
		#print("Admitted   Patient " + str(self.bed_contents[j].id) + " Status = " + str(self.bed_contents[j].status) + " at time: " + str(env.now))

	# Advance forward one time step
	def pass_time(self, env):		
		self.available_doctors = self.doctors
		self.check_on_patients()
		self.update_patient(env)
		self.add_to_beds(env)

class patient(object):
	def __init__(self, env, status, consume, time_to_heal, arrival_time, id):
		self.env           = env
		self.status        = status
		self.consume       = consume
		self.time_to_heal  = time_to_heal
		self.time_with_doc = time_to_heal/4 #Using the source http://ugeskriftet.dk/files/scientific_article_files/2018-12/a4558.pdf
		self.arrival_time  = arrival_time
		self.id = id


class patient_generator(object):
	def __init__(self, env):
		self.env = env
		self.total_patients = 0

	def make_patient(self, env):
		pat_esi = self.get_status(env)
		pat_com = self.get_consume(env, pat_esi)
		pat_tim = self.get_time(env, pat_esi)
		new_pat = patient(env, pat_esi, pat_com, pat_tim, env.now, self.total_patients)
		self.total_patients += 1
		return (new_pat)

	# Patient esi status upon enter the hospital
	def get_status(self, env):	
		return choice([1,2,3,4,5], 1, ESI_CHANCE)[0]

	# Amount resources patient consumes during their visit
	def get_consume(self, env, esi):
		return choice([1,2,3], 1, ESI_CHANCE[esi-1])[0]

	# Time patient needs in bed to be discharged
	def get_time(self, env, esi):
		return (ESI_TIME[esi - 1][0] + ESI_TIME[esi - 1][1] * random.uniform(.5, 1.5))


def setup(env, num_doctors, num_beds, previous_patients):

	hospital = Hospital(env, num_doctors, num_beds)
	mafia = patient_generator(env) #JOKE

	# Simulation loop - each iteration is one minute
	while True:
		patient_chance = random.random()
		if(patient_chance < 0.0477495): #Corvallis has the odds of 0.0477495 each minute of a patient showing up to the emergency department.
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
# Sequence of program execution

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


''' TODO
# Some patient 5's will wait indefinitely as other 5's cut in line
# Multiple days: include patients that were there yesterday?
'''