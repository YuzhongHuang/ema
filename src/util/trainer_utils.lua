-- Yuzhong Huang yuzhong.huang@students.olin.edu
-- George Chen hc25@rice.edu

-- A trainer script for HMDB-51. This file contains a train() function,
-- which given a set of hyper-parameters, data location and network module, 
-- will perform training through back-propagation.

require 'optim'

require "./train_utils"
require "./data_utils"


function train(optimState, trainParams, paths, model, criterion)
    local optimState = optimState

    -- decoding train parameters
    local iteration = trainParams["iteration"]
    local frameNum = trainParams["frameNum"]
    local batchSize = trainParams["batchSize"]
    local imgSize = trainParams["imgSize"]
	
    for epoch=1,iteration do
        print('Current Epoch: '..epoch)

        local parameters, gradParams = model:getParameters()

        -- call getBatch() to generate batchInputs and batchLabels
        local batchInputs, batchLabels = getBatch(paths["train"], paths["video"], batchSize, frameNum, imgSize)

        local function feval(params)
            -- get new parameters
            if params ~= parameters then
                parameters:copy(params)
            end

            -- reset gradients
            gradParams:zero()

            local outputs = model:forward(batchInputs)
            local loss = criterion:forward(outputs, batchLabels)
            print('Train Error '..loss)	

            local dloss_doutput = criterion:backward(outputs, batchLabels)
            model:backward(batchInputs, dloss_doutput)

            return loss,gradParams
        end

        optim.sgd(feval, parameters, optimState)

        model:zeroGradParameters() 
        model:forget()
    end

    -- clear model state to minimize memory
    model:clearState()
    -- save the model
    torch.save('model.t7', model)

    return model
end
