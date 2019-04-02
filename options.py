import os

class Options:
	def __init__(self, name = 'test', 
		buffer_size = 100000,
		batch_size=32, 
		gamma=0.99,
		tau=0.001, 
		lr=0.0005, 
		update_every=4, 
		conv=False, 
		double=False, 
		duel=False, 
		priority=False, 
		test=False, 
		env='LunarLander-v0', 
		seed=0,
		episodes = 10000,
	    steps = 1000,
	    eps_start = 1.0,
	    eps_end = 0.1, 
	    eps_decay = 0.995,
	    success_score = 200):

		self.name = name 					# string | name of learning session
		self.buffer_size = buffer_size      # int    | maximum capacity of memory replays
		self.batch_size = batch_size		# int    | size of experience batches used to train
		self.gamma = gamma					# float  | discount factor
		self.tau = tau 						# float  | interpolation factor (for updating network)
		self.lr = lr 						# float  | learning rate
		self.update_every = update_every 	# int    | steps between each model update
		self.conv = conv					# bool   | flag for the usage of conv networks (ie state is an image)
		self.double = double				# bool   | flag for usage of double dqn
		self.duel = duel 					# bool   | flag for usage of duel dqn
		self.priority = priority 			# bool   | flag for usage of priority experience replay
		self.test = test 					# bool   | flag to run test
		self.env = env 						# bool   | environment
		self.seed = seed					# bool   | random seeding
		self.episodes = episodes 			# bool   | number of episodes
		self.steps = steps 					# bool   | number of steps per episode
		self.eps_start = eps_start 			# bool   | epsilon at the beginning (for episilon-greedy selection)
		self.eps_end = eps_end 				# bool   | epsilon at the end (ie minimum eps)
		self.eps_decay = eps_decay			# bool   | rate of eps decay
		self.success_score = success_score  # bool   | score required for sucess

	def __str__(self):
		return str(self.__dict__)

	def save_as_txt(self):
		dir = os.path.join('./checkpoints', self.name)
		if not os.path.exists(dir):
			os.makedirs(dir)

		with open(os.path.join(dir, 'options.txt'), 'w+') as file:
			for key, value in self.__dict__.items():
				file.write("{0}: {1}\n".format(key,value))
