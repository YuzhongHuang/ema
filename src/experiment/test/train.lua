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

--torch.setdefaulttensortype('torch.FloatTensor') -- save some memory with FloatTensor data type

-- configurations
gpuFlag = true -- set running mode
imgSize = 16

trainPath = "../../../data/split1/train"
testPath = "../../../data/split1/test"
videoPath = "../../../data/frames"

-- hyper parameters
learningRate = 0.005 -- define the learning rate
learningDecay = 0.005
iteration = 200 -- define iteration num

-- parameters for building the network
frameNum = 20
channelNum = 3
classNum = 51
batchSize = 2 -- batchSize here is relative to each class. The actual batch size would be (batchSize) * (#classes)

-- model definition
rnn = learnable_ema(frameNum, channelNum, classNum, imgSize)
	:add(LRCN_nin_parallel(frameNum, channelNum*2, classNum, imgSize)):cuda()

-- initialize a parallel data table
net = nn.DataParallelTable(1)
net:add(rnn, {1,2,3,4,5,6,7,8})

-- build criterion
criterion = nn.ClassNLLCriterion()

-- transfer the net to gpu if gpu mode is on

net = net:cuda()
criterion = criterion:cuda()

-- Try to use optnet to reduce memory usage.
--input = torch.Tensor(classNum, frameNum, channelNum, imgSize, imgSize):zero()
--opts = {inplace=true, mode='training'}
--optnet = require 'optnet'
--optnet.optimizeMemory(rnn, input, opts)

--cudnn.convert(rnn, cudnn)
--cudnn.convert(criterion, cudnn)

a = train(iteration, learningRate, learningDecay, batchSize, frameNum, imgSize, net, criterion, trainPath, testPath, videoPath)
