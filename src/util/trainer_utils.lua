-- Yuzhong Huang yuzhong.huang@students.olin.edu
-- George Chen hc25@rice.edu

-- A trainer script for HMDB-51. This file contains a train() function,
-- which given a set of hyper-parameters, data location and network module, 
-- will perform training through back-propagation.

require 'optim'

require "./train_utils"
require "./data_utils"


function train(iterations, learningRate, learningDecay, batchSize, frameNum, imgSize, model, criterion, trainPath, testPath, videoPath)
    local optimState = {learningRate=learningRate, learningDecay=learningDecay, momentum = 0.5}

    -- call data_util.lua to generate test data
    -- ****Comment this block out for full dataset training*****
    local testset = getTest(testPath, videoPath, frameNum, batchSize, imgSize)
    local batchInputs, batchLabels = getBatch(trainPath, videoPath, batchSize, frameNum, imgSize)
    -- ****Comment this block out for full dataset training*****

    -- initialize an accuracy table to write down accuracies in each epoch 
    local accy_table = {}
	
    for epoch=1,iterations do
        print('Epoch '..epoch)

        local params, gradParams = model:getParameters()

        -- call data_util.lua to generate batchInputs and batchLabels
        
        -- ****UnComment this block out for full dataset training*****
        --local batchInputs, batchLabels = getBatch(trainPath, videoPath, batchSize, frameNum, imgSize)
	-- ****UnComment this block out for full dataset training*****

        local function feval(params)
            gradParams:zero()

            local outputs = model:forward(batchInputs)
            local loss = criterion:forward(outputs, batchLabels)
            print('Train Error '..loss)
            
	    -- ****UnComment this block out for full dataset training*****
            --local testset = getTest(testPath, videoPath, frameNum, batchSize, imgSize)
            -- ****UnComment this block out for full dataset training*****

            local accy = accuracy(model, testset)
            print("Test Accuracy "..accy)
            accy_table[epoch] = accy		

            local dloss_doutput = criterion:backward(outputs, batchLabels)
            model:backward(batchInputs, dloss_doutput)

            return loss,gradParams
        end

        optim.sgd(feval, params, optimState)

		-- call train_util.lua to compute the test accuracy
		-- local a = accuracy(model, testset)
		-- print('Test Accuracy '..a)

        model:zeroGradParameters() 
        model:forget()
    
        torch.save('accuracy.t7', accy_table)

    end

    -- save the model in the end of the training
    model:clearState()
    torch.save('model.t7', model)
    torch.save('accuracy.t7', accy_table)
end
