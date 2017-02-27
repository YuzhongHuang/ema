-- Yuzhong Huang yuzhong.huang@students.olin.edu
-- George Chen hc25@rice.edu
-- Train a network with viedo data and generate a plot of test accuracy  

require "gnuplot"
require "rnn"
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nn'

require "../../util/data_utils"
require "../../util/train_utils"
require "../../util/trainer_utils"
require "../../util/plot_utils"

-- configurations
gpuFlag = true -- set running mode
imgSize = 16
gpus = {1,2,3,4,5,6,7,8}

-- data paths
trainPath = "../../../data/split1/train"
testPath = "../../../data/split1/test"
videoPath = "../../../data/frames"

-- hyper parameters
learningRate = 0.005
learningDecay = 0.005
iteration = 200 -- #epochs
momentum = 0.5

-- parameters for building the network
frameNum = 20
channelNum = 3
classNum = 51
batchSize = 1 -- batchSize here is relative to each class. The actual batch size would be (batchSize) * (#classes)

-- encoding parameters into tables
paths = {train=trainPath, test=testPath, video=videoPath}
optimState = {learningRate=learningRate, learningDecay=learningDecay, momentum = momentum}
trainParams = {frameNum=frameNum, iteration=iteration, batchSize=batchSize, imgSize=imgSize}

-- generate a network model
rnn = learnable_ema(frameNum, channelNum, classNum, imgSize)
	:add(LRCN_margin_parallel(frameNum, channelNum*2, classNum, imgSize)):cuda()

-- initialize a parallel data table for gpu
if gpus then
	net = nn.DataParallelTable(1)
	net:add(rnn, gpus)
else
	net = rnn
end

-- build criterion
criterion = nn.ClassNLLCriterion()

-- transfer the net to gpu if gpu mode is on
if gpuFlag then
	net = net:cuda()
	criterion = criterion:cuda()
end

-- TODO: use optnet to reduce memory usage
-- TODO: use cudnn to optimize

-- call training function
trained_model = train(optimState, trainParams, paths, net, criterion)
-- test trained model with test dataset
accuracy(trained_model, getTest(paths["test"], paths["video"], frameNum, batchSize, imgSize))
