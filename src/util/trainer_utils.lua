-- Yuzhong Huang yuzhong.huang@students.olin.edu
-- George Chen hc25@rice.edu

-- A trainer script for HMDB-51. This file contains a train() function,
-- which given a set of hyper-parameters, data location and network module, 
-- will perform training through back-propagation.

require 'optim'

require "./train_utils"
require "./data_utils"


function train(optimState, opt, path, model, criterion)
    local optimState = optimState

    -- load a testset
    local testset = getTest(path.testPath, path.videoPath, opt.frameNum, opt.imgSize, opt.channelNum, opt.testBatchTotal, path.testName)
    
    -- epoch loop
    for epoch = 1, opt.iteration do
        print('Current Epoch: '..epoch)

        -- load a shuffled trainset
        trainset = {}
        trainset.paths, trainset.labels = getDataPath(path.trainPath, path.videoPath, opt.frameNum, opt.imgSize, opt.trainBatchTotal, path.trainName)

        local parameters, gradParams = model:getParameters()
        local epochError = 0

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

            -- convert labels to cuda tensors
            targets = torch.Tensor(targets):cuda()

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
                model:forget()

                -- get batch input from batch paths
                local input = getVideo(paths, opt.frameNum, opt.imgSize, opt.channelNum)

                -- evaluate function for complete mini batch
                local outputs = model:forward(input)
                local f = criterion:forward(outputs, targets)

                -- estimate df/dW
                local df_do = criterion:backward(outputs, targets)
                model:backward(input, df_do)

                -- TODO: consider using L1 and L2 penalties
                print(model.modules[1].modules[4].modules[2].modules[3].modules[1].modules[2].modules[1].weight)

                print("Batch error: "..f)
                epochError = epochError + f

                -- return f and df/dX
                return f, gradParams
            end
            optim.sgd(feval, parameters, optimState)
        end 

        --update epoch error
        epochError = epochError*opt.batchSize/(#(trainset.paths))
        print('Epoch error: '.. epochError)       

        print(accuracy(model, testset))
    end

    -- clear model state to minimize memory
    model:clearState()
    -- save the model
    torch.save("./models/model"..opt.exp_name..".t7", model)

    return model
end
