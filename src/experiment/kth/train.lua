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
gpus = {}

-- data loading path
trainPath = "../../../kthData/split1/train"
testPath = "../../../kthData/split1/test"
videoPath = "../../../kthData/frames"

-- hyper parameters
learningRate = 0.005
learningDecay = 0.005
iteration = 20 -- #epochs
momentum = 0.5

-- parameters for building the network
frameNum = 40
channelNum = 3
classNum = 6
relativeBatchSize = 10 -- batchSize here is relative to each class. The actual batch size would be (batchSize) * (#classes)
batchSize = relativeBatchSize * classNum

-- get the train and test dataset's paths and labels
trainset = {}
testset = {}
trainset.paths, trainset.labels = getEpoch(trainPath, videoPath, frameNum, imgSize)
testset.paths, testset.labels = getEpoch(testPath, videoPath, frameNum, imgSize)

-- encoding parameters into tables
optimState = {learningRate=learningRate, learningDecay=learningDecay, momentum = momentum}
opt = {frameNum=frameNum, iteration=iteration, batchSize=batchSize, imgSize=imgSize}

-- generate a network model
rnn = learnable_ema(frameNum, channelNum, classNum, imgSize)
	:add(LRCN_nin_parallel(frameNum, channelNum*2, classNum, imgSize)):cuda()

-- initialize a parallel data table for gpu
if next(gpus) == nil then
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
trained_model = train(optimState, opt, trainset, net, criterion)
-- test trained model with test dataset
accuracy(trained_model, getTest(paths.test, paths.video, frameNum, relativeBatchSize, imgSize))
