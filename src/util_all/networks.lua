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
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(Lenet(channelNum, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

    return model
end

-- expriment number 2: frame-driven lenet RCCN
function exp_2_no_fc(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(Lenet(channelNum, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel_no_fc(classNum, kernelNum, rnnSize))

    return model
end

function exp_2_no_fc_vgg(frameNum, channelNum, classNum, size)
    local kernelNum = 512
    local kernelSize = 7
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(Lenet(channelNum, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel_no_fc(classNum, kernelNum, rnnSize))

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
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = 2*kernelSize

    local model = nn.Sequential()
        :add(Lenet(channelNum, size))
        :add(Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

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

-- expriment number 5_m: multi-event-driven lenet LRCN
function exp_5_m(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(multi_ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*6, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Long_Term_Recurrent_lenet(frameNum, classNum, kernelNum, rnnSize))

    return model
end

-- expriment number 6: event-driven lenet RCCN
function exp_6(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*2, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

    return model
end

-- expriment number 6: event-driven lenet RCCN
function exp_6_no_fc(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*2, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel_no_fc(classNum, kernelNum, rnnSize))

    return model
end

-- expriment number 6: event-driven lenet RCCN
function exp_6_no_fc_two_stream(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local rcn1 = nn.Sequential()
	:add(nn.Contiguous())
	:add(Lenet(channelNum, size))
	:add(Non_Marginal(frameNum, kernelSize))
	:add(Recurrent_Per_Channel_no_fc(classNum, kernelNum, rnnSize))

    local rcn2 = nn.Sequential()
        :add(nn.Contiguous())
	:add(Lenet(channelNum, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel_no_fc(classNum, kernelNum, rnnSize))

    local model = nn.Sequential()
        :add(ema(frameNum, channelNum, classNum, size))
	:add(nn.SplitTable(2,4))
	:add(nn.ParallelTable()
	    :add(rcn1)
	    :add(rcn2))
	:add(nn.JoinTable(2,2))

    return model
end

-- expriment number 6: event-driven lenet RCCN
function exp_6_no_fc_vgg(frameNum, channelNum, classNum, size)
    local kernelNum = 512
    local kernelSize = 7
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(ema(frameNum, channelNum, classNum, size))
        :add(Vgg_19(channelNum*2, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel_no_fc(classNum, kernelNum, rnnSize))

    return model
end

function exp_frame_edr(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
	:add(nn.ConcatTable()
	    :add(exp_2_no_fc(frameNum, channelNum, classNum, size))
	    :add(exp_6_no_fc(frameNum, channelNum, classNum, size)))
	:add(nn.JoinTable(2,2))
	:add(nn.Linear(2*kernelNum*classNum, classNum))
	:add(nn.LogSoftMax())

    return model
end

function exp_frame_edr_three_stream(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(nn.ConcatTable()
            :add(exp_2_no_fc(frameNum, channelNum, classNum, size))
            :add(exp_6_no_fc_two_stream(frameNum, channelNum, classNum, size)))
        :add(nn.JoinTable(2,2))
        :add(nn.Linear(3*kernelNum*classNum, classNum))
        :add(nn.LogSoftMax())

    return model
end

function exp_frame_edr_vgg(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(nn.ConcatTable()
            :add(exp_2_no_fc_vgg(frameNum, channelNum, classNum, size))
            :add(exp_6_no_fc_vgg(frameNum, channelNum, classNum, size)))
        :add(nn.JoinTable(2,2))
        :add(nn.Linear(2*kernelNum*classNum, classNum))
        :add(nn.LogSoftMax())

    return model
end

-- expriment number 6_m: multi-event-driven lenet RCCN
function exp_6_m(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(multi_ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*6, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

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

-- expriment number 7_m: multi-event-driven lenet marginal LRCN
function exp_7_m(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = 2*kernelSize

    local model = nn.Sequential()
        :add(multi_ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*6, size))
        :add(Marginal(frameNum, kernelSize))
        :add(Long_Term_Recurrent_lenet(frameNum, classNum, kernelNum, rnnSize))

    return model
end

function exp_8_frame_and_ema(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = 2*kernelSize

    local model = nn.Sequential()
        :add(frame_and_ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*3, size))
        :add(Marginal(frameNum, kernelSize))
        :add(Long_Term_Recurrent_lenet(frameNum, classNum, kernelNum, rnnSize))

    return model
end

-- expriment number 8: event-driven lenet marginal RCCN
function exp_8(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = 2*kernelSize

    local model = nn.Sequential()
        :add(ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*2, size))
        :add(Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

    return model
end

-- expriment number 8_m: multi-event-driven lenet marginal RCCN
function exp_8_m(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = 2*kernelSize

    local model = nn.Sequential()
        :add(multi_ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*6, size))
        :add(Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

    return model
end

-- -- expriment number 9: frame-driven NiN LRCN
-- function exp_9(frameNum, channelNum, classNum, size)

-- end

-- expriment number 10: frame-driven NiN RCCN
function exp_10(frameNum, channelNum, classNum, size)
    local kernelNum = 192
    local kernelSize = size/4
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(NiN(channelNum, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

    return model
end

-- -- expriment number 11: frame-driven NiN marginal LRCN
-- function exp_11(frameNum, channelNum, classNum, size)

-- end

-- expriment number 12: frame-driven NiN marginal RCCN
function exp_12(frameNum, channelNum, classNum, size)
    local kernelNum = 192
    local kernelSize = size/4
    local rnnSize = 2*kernelSize

    local model = nn.Sequential()
        :add(NiN(channelNum, size))
        :add(Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

    return model
end

-- -- expriment number 13: event-driven NiN LRCN
-- function exp_13(frameNum, channelNum, classNum, size)

-- end

-- expriment number 14: event-driven NiN RCCN
function exp_14(frameNum, channelNum, classNum, size)
    local kernelNum = 192
    local kernelSize = size/4
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(ema(frameNum, channelNum, classNum, size))
        :add(NiN(channelNum*2, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

    return model
end

-- expriment number 14_m: multi-event-driven NiN RCCN
function exp_14_m(frameNum, channelNum, classNum, size)
    local kernelNum = 192
    local kernelSize = size/4
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(multi_ema(frameNum, channelNum, classNum, size))
        :add(NiN(channelNum*6, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

    return model
end

-- -- expriment number 15: event-driven NiN marginal LRCN
-- function exp_15(frameNum, channelNum, classNum, size)

-- end

-- expriment number 16: event-driven NiN marginal RCCN
function exp_16(frameNum, channelNum, classNum, size)
    local kernelNum = 192
    local kernelSize = size/4
    local rnnSize = 2*kernelSize

    local model = nn.Sequential()
        :add(ema(frameNum, channelNum, classNum, size))
        :add(NiN(channelNum*2, size))
        :add(Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

    return model
end

-- expriment number 16: event-driven NiN marginal RCCN
function exp_16_frame_and_ema(frameNum, channelNum, classNum, size)
    local kernelNum = 192
    local kernelSize = size/4
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(frame_and_ema(frameNum, channelNum, classNum, size))
        :add(NiN(channelNum*3, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

    return model
end

-- expriment number 16_m: multi-event-driven NiN marginal RCCN
function exp_16_m(frameNum, channelNum, classNum, size)
    local kernelNum = 192
    local kernelSize = size/4
    local rnnSize = 2*kernelSize

    local model = nn.Sequential()
        :add(ema(frameNum, channelNum, classNum, size))
        :add(ResNet(channelNum*2, size))
        :add(Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

    return model
end

-- experiment with different emas: texture ema
function exp_text_ema(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*2, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))
end

-- experiment with different emas: binary ema
function exp_bin_ema(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(bin_ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*2, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

    return model
end

-- experiment with different emas: multiple ema
function exp_multi_ema(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(multi_ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*2, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

    return model
end

-- experiment with different emas: binary ema
function exp_beta_ema(frameNum, channelNum, classNum, size)
    local kernelNum = 16
    local kernelSize = size/4 - 3
    local rnnSize = kernelSize*kernelSize

    local model = nn.Sequential()
        :add(bin_beta_ema(frameNum, channelNum, classNum, size))
        :add(Lenet(channelNum*2, size))
        :add(Non_Marginal(frameNum, kernelSize))
        :add(Recurrent_Per_Channel(classNum, kernelNum, rnnSize))

    return model
end
