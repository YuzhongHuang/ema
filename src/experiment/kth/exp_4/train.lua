-- Yuzhong Huang yuzhong.huang@students.olin.edu
-- George Chen hc25@rice.edu
-- Train a network with viedo data and generate a plot of test accuracy  

require "gnuplot"
require "rnn"
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nn'

require "../../../util/data_utils"
require "../../../util/train_utils"
require "../../../util/trainer_utils"
require "../../../util/plot_utils"
require "../../../util/modules"
require "../../../util/networks"

-- configurations
gpuFlag = true  -- set running mode
imgSize = 32
gpus = nil

-- data loading path
trainPath = "../../../../kthData/split1/train"
testPath = "../../../../kthData/split1/test"
videoPath = "../../../../kthData/frames"

trainName = "/train.txt"    -- name of the train split file
testName = "/test.txt"      -- name of the test split file

-- encoding path datas
path = {trainPath=trainPath, testPath=testPath, videoPath=videoPath, trainName=trainName, testName=testName}

-- data parameters
trainBatchTotal = 75
testBatchTotal = 24

-- hyper parameters
learningRate = 0.09
learningDecay = 0.008
iteration = 3  -- #epochs
momentum = 0.5

-- parameters for building the network
frameNum = 80
channelNum = 1
classNum = 6
relativeBatchSize = 5   -- batchSize here is relative to each class. The actual batch size would be (batchSize) * (#classes)
batchSize = relativeBatchSize * classNum

-- name of the experiment
exp_name = "_"..imgSize.."_"..learningRate.."_"..learningDecay.."_"..iteration.."_"..momentum.."_"..frameNum.."_"..batchSize

-- encoding parameters into tables
optimState = {
    learningRate = learningRate, 
    learningDecay = learningDecay, 
    momentum = momentum
}

opt = {
    frameNum = frameNum, 
    iteration = iteration, 
    batchSize = batchSize, 
    imgSize = imgSize, 
    channelNum = channelNum, 
    trainBatchTotal = trainBatchTotal, 
    testBatchTotal = testBatchTotal, 
    exp_name = exp_name
}

-- generate a network model
model = exp_4(frameNum, channelNum, classNum, imgSize):cuda()

-- initialize a parallel data table for gpu
if gpus ~= nil then
    net = nn.DataParallelTable(1)
    net:add(model, gpus)
else
    net = model
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
trained_model = train(optimState, opt, path, net, criterion)
