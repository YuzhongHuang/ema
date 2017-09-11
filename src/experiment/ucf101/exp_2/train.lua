-- Yuzhong Huang yuzhong.huang@students.olin.edu
-- George Chen hc25@rice.edu
-- Train a network with viedo data and generate a plot of test accuracy  

require "gnuplot"
require "rnn"
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nn'
require 'lfs'

require "../../../util_all/data_utils"
require "../../../util_all/train_utils"
require "../../../util_all/trainer_utils"
require "../../../util_all/plot_utils"
require "../../../util_all/modules"
require "../../../util_all/networks"

-- configurations
gpuFlag = true  -- set running mode
imgSize = 32
gpus = {1,2,3,4,5,6,7,8}

-- data loading path
trainPath = "/mnt/group2/crnn/ucf101Data/split1/train"
testPath = "/mnt/group2/crnn/ucf101Data/split1/test"
videoPath = "/mnt/group2/crnn/ucf101Data/frames"
load_or_save = "load"
train_file = "ucf101Train.t7"
test_file = "ucf101Test.t7"

trainName = "/train.txt"    -- name of the train split file
testName = "/test.txt"      -- name of the test split file

-- encoding path datas
path = {trainPath=trainPath, testPath=testPath, videoPath=videoPath, trainName=trainName, testName=testName}

-- data parameters
trainBatchTotal = -1
testBatchTotal = -1

-- hyper parameters
learningRate = 1.2
learningDecay = 0.01
iteration = 200  -- #epochs
momentum = 0.5

-- parameters for building the network
frameNum = 80
channelNum = 1
classNum = 101
relativeBatchSize = 2   -- batchSize here is relative to each class. The actual batch size would be (batchSize) * (#classes)
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
model = exp_2(frameNum, channelNum, classNum, imgSize):cuda()

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
trained_model = train(optimState, opt, path, net, criterion, load_or_save, train_file, test_file)
