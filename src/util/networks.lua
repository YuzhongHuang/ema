-- Yuzhong Huang yuzhong.huang@students.olin.edu

-- This script contains network code for our experiments settings

-- import modules code
require './modules'

-- expriment number 1: frame-driven lenet LRCN
function exp_1(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(Lenet(channelNum, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Long_Term_Recurrent_lenet(frameNum, classNum, kernelNum, rnnSize))

    return model
end

-- expriment number 2: frame-driven lenet RCCN
function exp_2(frameNum, channelNum, classNum, size)
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(Lenet(channelNum, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, rnnSize))

    return model
end

-- expriment number 3: frame-driven lenet marginal LRCN
function exp_3(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = 2*kernelSize

    local model = nn.Sequential()
        :add(Lenet(channelNum, size))
        :add(Marginal(frameNum, kernelSize))
        :add(Long_Term_Recurrent_lenet(frameNum, classNum, kernelNum, rnnSize))

    return model
end

-- expriment number 4: frame-driven lenet marginal RCCN
function exp_4(frameNum, channelNum, classNum, size)
    local kernelSize = size/4 - 3
    local rnnSize = 2*kernelSize

    local model = nn.Sequential()
        :add(Lenet(channelNum, size))
        :add(Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, rnnSize))

    return model
end

-- expriment number 5: event-driven lenet LRCN
function exp_5(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*2, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Long_Term_Recurrent_lenet(frameNum, classNum, kernelNum, rnnSize))

    return model
end

-- expriment number 6: event-driven lenet RCCN
function exp_6(frameNum, channelNum, classNum, size)
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*2, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, rnnSize))

    return model
end

-- expriment number 7: event-driven lenet marginal LRCN
function exp_7(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = 2*kernelSize

    local model = nn.Sequential()
        :add(ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*2, size))
        :add(Marginal(frameNum, kernelSize))
        :add(Long_Term_Recurrent_lenet(frameNum, classNum, kernelNum, rnnSize))

    return model
end

-- expriment number 8: event-driven lenet marginal RCCN
function exp_8(frameNum, channelNum, classNum, size)
    local kernelSize = size/4 - 3
    local rnnSize = 2*kernelSize

    local model = nn.Sequential()
        :add(ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*2, size))
        :add(Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, rnnSize))

    return model
end

-- -- expriment number 9: frame-driven NiN LRCN
-- function exp_9(frameNum, channelNum, classNum, size)

-- end

-- expriment number 10: frame-driven NiN RCCN
function exp_10(frameNum, channelNum, classNum, size)
    local kernelSize = size/4
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(NiN(channelNum, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, rnnSize))

    return model
end

-- -- expriment number 11: frame-driven NiN marginal LRCN
-- function exp_11(frameNum, channelNum, classNum, size)

-- end

-- expriment number 12: frame-driven NiN marginal RCCN
function exp_12(frameNum, channelNum, classNum, size)
    local kernelSize = size/4
    local rnnSize = 2*kernelSize

    local model = nn.Sequential()
        :add(NiN(channelNum, size))
        :add(Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, rnnSize))

    return model
end

-- -- expriment number 13: event-driven NiN LRCN
-- function exp_13(frameNum, channelNum, classNum, size)

-- end

-- expriment number 14: event-driven NiN RCCN
function exp_14(frameNum, channelNum, classNum, size)
    local kernelSize = size/4
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(ema(frameNum, channelNum, classNum, size))
        :add(NiN(channelNum*2, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, rnnSize))

    return model
end

-- -- expriment number 15: event-driven NiN marginal LRCN
-- function exp_15(frameNum, channelNum, classNum, size)

-- end

-- expriment number 16: event-driven NiN marginal RCCN
function exp_16(frameNum, channelNum, classNum, size)
    local kernelSize = size/4
    local rnnSize = 2*kernelSize

    local model = nn.Sequential()
        :add(ema(frameNum, channelNum, classNum, size))
        :add(NiN(channelNum*2, size))
        :add(Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, rnnSize))

    return model
end
