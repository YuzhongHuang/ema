-- Yuzhong Huang yuzhong.huang@students.olin.edu
-- George Chen hc25@rice.edu

-- A util stcript for HMDB-51. This script contains all the
-- network modules, as well as test data validation function

-- accuracy() computes the test accuracy of a given test data
function accuracy(net, testData)
    local correct = 0
    local test_num = (#testData.vids)[1]

    local groundtruths = testData.labels
    local predictions = net:forward(testData.vids)

    for i=1,test_num do
        local groundtruth = groundtruths[i]
        local prediction = predictions[i]

        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            correct = correct + 1
        end
    end

    return correct*(100.0/test_num) -- convert the correctness to percentages
end

function learnable_ema(frameNum, channelNum, classNum, size)
    -- set up the EMA
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

    local net = nn.Sequential()
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
        :add(nn.MulConstant(50))        
        :add(nn.View(frameNum, channelNum*2, size, size))
    return net
end

-- function that returns an ResNet with marginalization and channel based GRU
function LRCN_ResNet_parallel_32(frameNum, channelNum, classNum, size)
    -- tensor in: 40x3x32x32
    require 'cunn' -- frameNum = 40; channelNum = 3; classNum = 51; size = 32
    
    local model = nn.Sequential()
    model:add(nn.View(channelNum, size, size))

    local resnet = torch.load('cifar10_32x32_20.t7')
    resnet:remove(9) -- Linear fcn 64 -> 10
    resnet:remove(8) -- View 
    resnet:remove(7)-- Spatial average pooling 8x8 -> 1x1

    model:add(resnet):cuda() -- size:40x64x8x8

    model:add(nn.View(64,8,8)) --40x8x8x64 was the ouput from resnet
    model:add(nn.Replicate(1,2)) -- size:40x1x64x8x8
    model:add(nn.ConcatTable() --marginalization
        :add(nn.Sum(4)) -- sum over x - size: 40x1x64x8
        :add(nn.Sum(5))) -- sum over y - size: 40x1x64x8
    model:add(nn.JoinTable(2)) -- become 40x2x64x8
    model:add(nn.ReLU()) -- 40x2x64x8

    model:add(nn.View(frameNum, 8)) -- size: 128x40x8
    model:add(nn.SplitTable(2,3)) -- size: 40 tables of 128x8
    model:add(nn.Sequencer(nn.LSTM(8,classNum,5))) -- size: 40 tables of 128x51
    model:add(nn.SelectTable(-1)) -- size: 128x51

    model:add(nn.View(2 * 64 * 51)) -- 
    model:add(nn.Linear(2 * 64 * 51, 51))
    model:add(nn.LogSoftMax())

    model:cuda() -- model needs to be cuda 
    -- also, tensor needs to be cuda'd: model_out = model:forward(batch:cuda())


    return model
end

-- function that returns an Lenet with marginalization and channel based GRU
function LRCN_margin_parallel(frameNum, channelNum, classNum, size)
    local conv_size = size/4 - 3

    local net = nn.Sequential()
       :add(nn.View(channelNum, size, size))
       :add(nn.SpatialConvolution(channelNum, 6, 5, 5))
       :add(nn.SpatialBatchNormalization(6))
       :add(nn.ReLU()) 
       :add(nn.SpatialMaxPooling(2,2,2,2))
       :add(nn.SpatialConvolution(6, 16, 5, 5))
       :add(nn.SpatialBatchNormalization(16))
       :add(nn.ReLU()) 
       :add(nn.SpatialMaxPooling(2,2,2,2))
       :add(nn.Replicate(1,2))
       :add(nn.ConcatTable()
            :add(nn.Sum(4))
            :add(nn.Sum(5)))
       :add(nn.JoinTable(2))
       :add(nn.ReLU())
       :add(nn.View(frameNum, conv_size)) 
       :add(nn.SplitTable(2,3))
       :add(nn.Sequencer(nn.GRU(conv_size, classNum)))
       :add(nn.SelectTable(-1))
       :add(nn.View(2*16*classNum))
       :add(nn.Linear(2*16*classNum, classNum))
       :add(nn.LogSoftMax())
       
    return net
end

-- function that returns an NiN with marginalization and channel based GRU
function LRCN_nin_parallel(frameNum, channelNum, classNum, size)
    local model = nn.Sequential()
    model:add(nn.View(channelNum, size, size))

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

    model:add(nn.View(192,size/4,size/4))
    model:add(nn.Replicate(1,2))
    model:add(nn.ConcatTable()
        :add(nn.Sum(4))
        :add(nn.Sum(5)))
    model:add(nn.JoinTable(2))
    model:add(nn.ReLU())

    model:add(nn.View(frameNum, size/4))
    model:add(nn.SplitTable(2,3))
    model:add(nn.Sequencer(nn.LSTM(size/4,classNum,5)))
    model:add(nn.SelectTable(-1))

    model:add(nn.View(2*192*classNum))
    model:add(nn.Linear(2*192*classNum, classNum))
    model:add(nn.LogSoftMax())

    return model
end



function LRCN_ResNet_parallel_256(frameNum, channelNum, classNum, size)
    require 'cunn'

    local model = nn.Sequential()
    model:add(nn.View(channelNum, size, size))

    local resnet = torch.load('resnet-34.t7')
    resnet:remove(9) -- Linear fcn 1000 -> 512
    resnet:remove(8) -- Spatial average pooling 7x7 -> 1x1


    model:add(resnet)	

    model:add(nn.View(64,49,49)) --2401(7^2x7^2)x64 was the ouput from resnet
    model:add(nn.Replicate(1,2))
    model:add(nn.ConcatTable()
        :add(nn.Sum(4))
        :add(nn.Sum(5)))
    model:add(nn.JoinTable(2))
    model:add(nn.ReLU())

    model:add(nn.View(frameNum, 49))
    model:add(nn.SplitTable(2,3))
    model:add(nn.Sequencer(nn.LSTM(size/4,classNum,5)))
    model:add(nn.SelectTable(-1))

    model:add(nn.View(2*192*6))
    model:add(nn.Linear(2*192*6, 6))
    model:add(nn.LogSoftMax())

    model:cuda() -- model needs to be cuda 
    -- also, tensor needs to be cuda'd: model_out = model:forward(batch:cuda())

    return model
end

-- -- EMA layer that can be used on top of CNN
-- function learnable_ema(batchSize, channelNum, classNum, size)
--     -- set up the EMA
--     local input_layer = nn.Sequential():add(nn.Mul()):add(nn.Abs())
--     local hidden_layer_alpha = input_layer:clone('weight','bias','gradWeight','gradBias')
        
--     local hidden_layer = nn.Sequential()
--         :add(nn.ConcatTable()
--             :add(nn.Identity())
--             :add(hidden_layer_alpha))
--         :add(nn.CSubTable())
    
--     local r = nn.Recurrent(nn.Abs(), input_layer, hidden_layer, nn.Abs(), 5)
    
--     local ema = nn.Sequential()
--         :add(nn.AddConstant(1))
--         :add(nn.SplitTable(1,2))
--         :add(nn.Sequencer(r))
--         :add(nn.JoinTable(2))
    
--     local identity = nn.Sequential()
--         :add(nn.AddConstant(1))
    
--     local mean = nn.Sequential()
--         :add(nn.Mean(3))
--         :add(nn.Replicate(channelNum*size*size,3))
    
--     local frame_normalize = nn.Sequential()
--         :add(nn.ConcatTable()
--             :add(nn.Identity())
--             :add(mean))
--         :add(nn.CSubTable())
    
--     local pos = nn.Sequential()
--         :add(nn.AddConstant(-0.003))
--         :add(nn.ReLU())
    
--     local neg = nn.Sequential()
--         :add(nn.MulConstant(-1))
--         :add(nn.AddConstant(-0.003))
--         :add(nn.ReLU())

--     local net = nn.Sequential()
--         :add(nn.View(batchSize, channelNum*size*size))
--         :add(nn.Transpose({1,2}))
--         :add(frame_normalize)
--         :add(nn.Tanh())
--         :add(nn.ConcatTable()
--             :add(identity)
--             :add(ema))
--         :add(nn.CDivTable())
--         :add(nn.View(batchSize, channelNum*size*size))
--         :add(nn.Transpose({1,2}))
--         :add(frame_normalize)
--         :add(nn.View(batchSize, channelNum, size*size))
--         :add(nn.Transpose({1,2}))
--         :add(nn.Tanh())
--         :add(nn.ConcatTable()
--             :add(pos)
--             :add(neg))
--         :add(nn.JoinTable(3))
--         :add(nn.MulConstant(20))        
--         :add(nn.View(batchSize, channelNum*2, size, size))
--         :add(nn.Transpose({1,2}))
--     return net
-- end

-- -- function that returns an Lenet with marginalization and channel based GRU
-- function LRCN_margin_parallel(FrameNum, channelNum, classNum, imgSize)
--     local net = nn.Sequential()
--        :add(nn.View(channelNum, imgSize, imgSize))
--        :add(nn.SpatialConvolution(channelNum, 6, 5, 5))
--        :add(nn.SpatialBatchNormalization(6))
--        :add(nn.ReLU()) 
--        :add(nn.SpatialMaxPooling(2,2,2,2))
--        :add(nn.SpatialConvolution(6, 16, 5, 5))
--        :add(nn.SpatialBatchNormalization(16))
--        :add(nn.ReLU()) 
--        :add(nn.SpatialMaxPooling(2,2,2,2))
--        :add(nn.Replicate(1,2))
--        :add(nn.ConcatTable()
--             :add(nn.Sum(4))
--             :add(nn.Sum(5)))
--        :add(nn.JoinTable(2))
--        :add(nn.ReLU())
--        :add(nn.View(batchSize*2*16, (imgSize/4)-3))
--        :add(nn.Transpose({1,2}))
--        :add(nn.SplitTable(2,3))
--        :add(nn.Sequencer(nn.GRU((imgSize/4)-3, classNum)))
--        :add(nn.SelectTable(-1))
--        :add(nn.View(2*16*6))
--        :add(nn.Linear(2*16*6, classNum))
--        :add(nn.LogSoftMax())
       
--     return net
-- end

-- -- function that returns an NiN with marginalization and channel based GRU
-- function LRCN_nin_parallel(batchSize, channelNum, classNum, size)
--     local model = nn.Sequential()
--     model:add(nn.View(channelNum, size, size))

--     local function Block(...)
--         local arg = {...}
--         model:add(nn.SpatialConvolution(...))
--         model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
--         model:add(nn.ReLU(true))
--         return model
--     end

--     Block(channelNum,192,5,5,1,1,2,2)
--     Block(192,160,1,1)
--     Block(160,96,1,1)
--     model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
--     model:add(nn.Dropout())
--     Block(96,192,5,5,1,1,2,2)
--     Block(192,192,1,1)
--     Block(192,192,1,1)
--     model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
--     model:add(nn.Dropout())
--     Block(192,192,3,3,1,1,1,1)
--     Block(192,192,1,1)

--     model:add(nn.View(192,size/4,size/4))
--     model:add(nn.Replicate(1,2))
--     model:add(nn.ConcatTable()
--         :add(nn.Sum(4))
--         :add(nn.Sum(5)))
--     model:add(nn.JoinTable(2))
--     model:add(nn.ReLU())

--     model:add(nn.View(batchSize*192*2, size/4))
--     :add(nn.Transpose({1,2}))
--     model:add(nn.SplitTable(2,3))
--     model:add(nn.Sequencer(nn.LSTM(size/4,classNum,5)))
--     model:add(nn.SelectTable(-1))

--     model:add(nn.View(2*192*classNum))
--     model:add(nn.Linear(2*192*classNum, classNum))
--     model:add(nn.LogSoftMax())

--     return model
-- end
