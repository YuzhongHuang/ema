-- Yuzhong Huang yuzhong.huang@students.olin.edu
-- George Chen hc25@rice.edu

-- A trainer script for HMDB-51. This file contains a train() function,
-- which given a set of hyper-parameters, data location and network module, 
-- will perform training through back-propagation.

require 'optim'

require "./train_utils"
require "./data_utils"


function train(iterations, learningRate, learningDecay, batchSize, frameNum, imgSize, model, criterion, trainPath, testPath, videoPath)
    -- encoding high level parameters
    local optimState = {learningRate=learningRate, learningDecay=learningDecay, momentum = 0.5}

    -- call getTest() to generate test data
    local testset = getTest(testPath, videoPath, frameNum, batchSize, imgSize)

    -- initialize an accuracy table to write down accuracies in each epoch 
    local accuracy_table = {}
	
    for epoch=1,iterations do
        print('Current Epoch: '..epoch)

        local parameters, gradParams = model:getParameters()

        -- call getBatch() to generate batchInputs and batchLabels
        local batchInputs, batchLabels = getBatch(trainPath, videoPath, batchSize, frameNum, imgSize)

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

            local accuracy = accuracy(model, testset)
            print("Test Accuracy "..accuracy)
            accuracy_table[epoch] = accuracy		

            local dloss_doutput = criterion:backward(outputs, batchLabels)
            model:backward(batchInputs, dloss_doutput)

            return loss,gradParams
        end

        optim.sgd(feval, parameters, optimState)

		-- call train_util.lua to compute the test accuracy
		local a = accuracy(model, testset)
		print('Test Accuracy '..a)

        model:zeroGradParameters() 
        model:forget()
    
        torch.save('accuracy.t7', accuracy_table)
    end

    -- save the model in the end of the training
    model:clearState()
    torch.save('model.t7', model)
end
