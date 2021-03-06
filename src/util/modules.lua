-- Yuzhong Huang yuzhong.huang@students.olin.edu
-- George Chen hc25@rice.edu

-- This script contains all the modularized network code

-- ema network layer, with a tunable alpha value and a fixed threshold value
function ema(frameNum, channelNum, classNum, size)
    local input_layer = nn.Sequential():add(nn.Mul()):add(nn.Abs())
    local hidden_layer_alpha = input_layer:clone('weight','bias','gradWeight','gradBias')
        
    local hidden_layer = nn.Sequential()
        :add(nn.ConcatTable()
            :add(nn.Identity())
            :add(hidden_layer_alpha))
        :add(nn.CSubTable())
    
    local r = nn.Recurrent(nn.Abs(), input_layer, hidden_layer, nn.Abs(), 5)
    
    local ema = nn.Sequential()
        :add(nn.AddConstant(1))
        :add(nn.SplitTable(1,2))
        :add(nn.Sequencer(r))
        :add(nn.JoinTable(2))
    
    local identity = nn.Sequential()
        :add(nn.AddConstant(1))
    
    local mean = nn.Sequential()
        :add(nn.Mean(3))
        :add(nn.Replicate(channelNum*size*size,3))
    
    local frame_normalize = nn.Sequential()
        :add(nn.ConcatTable()
            :add(nn.Identity())
            :add(mean))
        :add(nn.CSubTable())
    
    local pos = nn.Sequential()
        :add(nn.AddConstant(-0.003))
        :add(nn.ReLU())
    
    local neg = nn.Sequential()
        :add(nn.MulConstant(-1))
        :add(nn.AddConstant(-0.003))
        :add(nn.ReLU())

    local model = nn.Sequential()
        :add(nn.View(frameNum, channelNum*size*size))
        :add(frame_normalize)
        :add(nn.Tanh())
        :add(nn.ConcatTable()
            :add(identity)
            :add(ema))
        :add(nn.CDivTable())
        :add(nn.View(frameNum, channelNum*size*size))
        :add(frame_normalize)
        :add(nn.View(frameNum, channelNum, size*size))
        :add(nn.Tanh())
        :add(nn.ConcatTable()
            :add(pos)
            :add(neg))
        :add(nn.JoinTable(3))
        :add(nn.MulConstant(20))        
        :add(nn.View(frameNum, channelNum*2, size, size))

    return model
end

-- Lenet
function Lenet(channelNum, size)
    local kernelSize = size/4 -3

    local model = nn.Sequential()
       :add(nn.View(channelNum, size, size))
       :add(nn.SpatialConvolution(channelNum, 6, 5, 5))
       :add(nn.SpatialBatchNormalization(6))
       :add(nn.ReLU()) 
       :add(nn.SpatialMaxPooling(2,2,2,2))
       :add(nn.SpatialConvolution(6, 16, 5, 5))
       :add(nn.SpatialBatchNormalization(16))
       :add(nn.ReLU()) 
       :add(nn.SpatialMaxPooling(2,2,2,2))
       :add(nn.View(16, kernelSize, kernelSize))

    return model
end

-- NiN
function NiN(channelNum, size)
    local kernelSize = size/4

    local model = nn.Sequential()
        :add(nn.View(channelNum, size, size))

    -- helper function to build NiN
    local function Block(...)
        local arg = {...}
        model:add(nn.SpatialConvolution(...))
        model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
        model:add(nn.ReLU(true))
        return model
    end

    Block(channelNum,192,5,5,1,1,2,2)
    Block(192,160,1,1)
    Block(160,96,1,1)
    model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
    model:add(nn.Dropout())
    Block(96,192,5,5,1,1,2,2)
    Block(192,192,1,1)
    Block(192,192,1,1)
    model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
    model:add(nn.Dropout())
    Block(192,192,3,3,1,1,1,1)
    Block(192,192,1,1)
    model:add(nn.View(192, kernelSize, kernelSize))

    return model
end

function Marginal(frameNum, kernelSize)
    local model = nn.Sequential()
        :add(nn.Replicate(1,2))
        :add(nn.ConcatTable()
            :add(nn.Sum(4))
            :add(nn.Sum(5)))
        :add(nn.JoinTable(2))
        :add(nn.ReLU())
        :add(nn.View(frameNum, 2*kernelSize))

    return model    
end

function Non_Marginal(frameNum, kernelSize)
    local model = nn.Sequential()
        :add(nn.View(frameNum, kernelSize*kernelSize))

    return model
end

function Recurrent_Per_Channel(classNum, kernelNum, rnnSize)
    local model = nn.Sequential()
        :add(nn.SplitTable(2,3))
        :add(nn.Sequencer(nn.LSTM(rnnSize,classNum)))
        :add(nn.SelectTable(-1))

        :add(nn.View(kernelNum*classNum))
        :add(nn.Linear(kernelNum*classNum, classNum))
        :add(nn.LogSoftMax())

    return model
end

function Long_Term_Recurrent_lenet(frameNum, classNum, kernelNum, rnnSize)
    local model = nn.Sequential()
        :add(nn.View(kernelNum*rnnSize))
        :add(nn.Linear(kernelNum*rnnSize, 120))
        :add(nn.View(frameNum, 120))

        :add(nn.SplitTable(2,3))
        :add(nn.Sequencer(nn.LSTM(120, 84)))
        :add(nn.SelectTable(-1))
        
        :add(nn.Linear(84, classNum))
        :add(nn.LogSoftMax())
    return model
end