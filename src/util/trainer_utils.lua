-- Yuzhong Huang yuzhong.huang@students.olin.edu
-- George Chen hc25@rice.edu

-- A trainer script for HMDB-51. This file contains a train() function,
-- which given a set of hyper-parameters, data location and network module, 
-- will perform training through back-propagation.

require 'optim'

require "./train_utils"
require "./data_utils"


function train(optimState, opt, trainset, model, criterion)
    local optimState = optimState
	
    -- epoch loop
    for epoch = 1, opt.iteration do
        print('Current Epoch: '..epoch)

        local parameters, gradParams = model:getParameters()

        -- loop through all the data with minibatches
        for t = 1, #(trainset.paths), opt.batchSize do
            print('Batch progress: '..t..'/'..#(trainset.paths))

            -- create minibatches
            local paths = {}
            local targets = {}

            for i = t, math.min(t+opt.batchSize-1, #(trainset.paths)) do
                -- load new sample
                local path = trainset.paths[i]
                local target = trainset.labels[i]
                table.insert(paths, path)
                table.insert(targets, target)
            end

            -- create closure to evaluate f(X) and df/dX

            local function feval(params)
                -- just in case:
                collectgarbage()

                -- get new parameters
                if params ~= parameters then
                    parameters:copy(params)
                end

                -- reset gradients
                gradParams:zero()

                -- get batch input from batch paths
                local input = getVideo(paths, opt.FrameNum, opt.imgSize)

                -- evaluate function for complete mini batch
                local outputs = model:forward(inputs)
                local f = criterion:forward(outputs, targets)

                -- estimate df/dW
                local df_do = criterion:backward(outputs, targets)
                model:backward(inputs, df_do)

                -- TODO: consider using L1 and L2 penalties

                -- return f and df/dX
                return f, gradParams
            end

            optim.sgd(feval, parameters, optimState)
            model:forget()
        end        
    end

    -- clear model state to minimize memory
    model:clearState()
    -- save the model
    torch.save('model.t7', model)

    return model
end