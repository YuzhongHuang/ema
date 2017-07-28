-- Yuzhong Huang yuzhong.huang@students.olin.edu
-- George Chen hc25@rice.edu

-- A trainer script for HMDB-51. This file contains a train() function,
-- which given a set of hyper-parameters, data location and network module, 
-- will perform training through back-propagation.

require 'optim'

require "./train_utils"
require "./data_utils"
require "./plot_utils"

math.randomseed(os.time())

function train(optimState, opt, path, model, criterion)
    -- initialize tables for recording trainning and testing result data
    epochErrors = {}
    accuracies = {}
    iterations = {}

    -- load a testset
    local testset = getTest(path.testPath, path.videoPath, opt.frameNum, opt.imgSize, opt.channelNum, opt.testBatchTotal, path.testName)
    
    -- load a trainset to cpu
    local trainset = getTrain(path.trainPath, path.videoPath, opt.frameNum, opt.imgSize, opt.channelNum, opt.trainBatchTotal, path.trainName)

    -- epoch loop
    for epoch = 1, opt.iteration do
        print('Current Epoch: '..epoch)

        local parameters, gradParams = model:getParameters()
        local epochError = 0
        local accuracy = 0

        -- loop through all the data with minibatches
        for t = 1, trainset.labels:size()[1], opt.batchSize do
            print('Batch progress: '..t..'/'..trainset.labels:size()[1])

            -- generate a random indices of whole trainset
            local indices = getIndices(trainset.labels:size()[1], trainset.labels:size()[1])

            local current_batchSize = math.min(t+opt.batchSize-1, trainset.labels:size()[1]) - t

            -- initialize batch input and targets
            local input = torch.FloatTensor(current_batchSize, opt.frameNum, opt.channelNum, opt.imgSize, opt.imgSize)
            local targets = {}

            -- load batch input and targets
            for i = 1, current_batchSize do
                local vid = trainset.vids[indices[i+t]]
                local vid_frameNum = (#vid)[1]  -- frame number of the vid
                local start = 1
                local frame_end = 1
                -- randomly choose a start for data augmentation
                if vid_frameNum  > opt.frameNum then
                    start = math.random(vid_frameNum - opt.frameNum)
                    frame_end = start + opt.frameNum - 1
                else
                    frame_end = vid_frameNum
                end
                -- load the vid to batch input
                input[{{i},{1, frame_end - start + 1},{},{},{}}] = vid[{{start, frame_end},{},{},{}}]
                -- insert corresponding targets
                table.insert(targets, trainset.labels[indices[i+t]])
            end

            -- convert labels to cuda tensors
            local targets = torch.Tensor(targets):cuda()

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

                -- evaluate function for complete mini batch
                local outputs = model:forward(input:cuda()) -- load batch input to gpu
                local f = criterion:forward(outputs, targets)

                -- estimate df/dW
                local df_do = criterion:backward(outputs, targets)
                model:backward(input, df_do)

                -- TODO: consider using L1 and L2 penalties
                --  print(model.modules[1].modules[4].modules[2].modules[3].modules[1].modules[2].modules[1].weight)

                print("Batch error: "..f)
                epochError = epochError + f

                -- return f and df/dX
                return f, gradParams
            end
            optim.sgd(feval, parameters, optimState)
        end 

        -- update and record epoch error
        epochError = epochError*opt.batchSize/(trainset.labels:size()[1])
        table.insert(epochErrors, epochError)

        print('Epoch Error: '..epochError)       

        -- update and record accuracy
        accuracy = getAccuracy(model, testset)
        table.insert(accuracies, accuracy)

        print('Test Accuracy: '..accuracy)

        -- update iterations
        table.insert(iterations, epoch)

        -- plot the train and test accuracy in realtime
        plot("./plots/plot"..opt.exp_name..".t7", iterations, accuracies, epochErrors)
    end
    
    model:clearState() -- clear model state to minimize memory
    torch.save("./models/model"..opt.exp_name..".t7", model) -- save the model

    -- save the train&test result data
    torch.save("./epochErrors/epochError"..opt.exp_name..".t7", epochErrors)
    torch.save("./accuracies/accuracy"..opt.exp_name..".t7", accuracies)

    return model
end
