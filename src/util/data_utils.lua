-- Yuzhong Huang yuzhong.huang@students.olin.edu
-- George Chen hc25@rice.edu

-- A data loading util stcript for HMDB-51. Given batch size, train\test data paths, 
-- splits txt files path, generates batch labels and batch inputs.

require 'cutorch'
require 'cunn'
require 'image'

math.randomseed(os.time())

-- getTest(path) return a table with a video tensor and a label tensor
function getTest(testpath, videoPath, frameNum, imgSize, channelNum, testBatchTotal, testName)
    local testSet = {}
    local paths = {}
    local labels = {}

    paths, labels = getDataPath(testPath, videoPath, frameNum, imgSize, testBatchTotal, testName)

    testSet.vids = getVideo(paths, frameNum, imgSize, channelNum)
    testSet.labels = torch.Tensor(labels):cuda()

    return testSet
end

function getDataPath(trainsets, videoPath, frameNum, imgSize, trainBatchTotal, fileName)
    local classes = ls(trainsets)
    local Paths = {}
    local Labels = {}

    for i=1, #classes do
        local path = trainsets..'/'..classes[i]..fileName
        local framePath = videoPath..'/'..classes[i]
        local lst = read_and_process(path, framePath)
        local indices = getIndices(trainBatchTotal, trainBatchTotal) -- get all shuffled elements per catergory for training

        -- get all trainning elements many videos from each class
        for j=1, trainBatchTotal do
            table.insert(Paths, lst[indices[j]])
            table.insert(Labels, i)
        end 
    end

    return Paths, Labels
end

function getVideo(paths, frameNum, imgSize, channelNum)
    local batchInputs = torch.FloatTensor(#paths, frameNum, channelNum, imgSize, imgSize)
    for i=1, #paths do
        local path = paths[i]

        -- deal with some formating issue with the file system
        sys_path = path:gsub('(%))', '\\%)')
        sys_path = path:gsub('(%()', '\\%(')
        sys_path = path:gsub('(%;)', '\\%;')
        sys_path = path:gsub('(&)', '\\&')
        
        -- get all img files under the current folder
        local frames = ls(sys_path)
        -- grab #frameNum of images porprotionally from #frames images
        local step = (#frames)/frameNum

        for j=1, frameNum do
            -- load a frame
            local index = math.floor(j*step+0.5)    -- return an integer closest to j*step 
            local frame = frames[index]

            -- load and resize the image
            local img = image.load(path..'/'..frame, channelNum, 'float')
            img = image.scale(img, imgSize, imgSize)

            -- store the images into the tensor
            batchInputs[{{i},{j},{},{},{}}] = img
        end
    end

    return batchInputs:cuda()
end


-- given a set of video paths, figure out the largest sequence number of a video
function maxSequence(paths)
    local max = 0

    for i=1, #paths do
        local path = paths[i]

        -- deal with some formating issue with the file system
        path = path:gsub('(%))', '\\%)')
        path = path:gsub('(%()', '\\%(')
        path = path:gsub('(%;)', '\\%;')
        path = path:gsub('(&)', '\\&')
        local frames = ls(path)

        if (#frames) > max then
            max = #frames
        end
    end

    return max
end

-- Define ls function
function ls(path) 
    return sys.split(sys.ls(path),'\n') 
end 

-- Read lines from file and process it by adding address prefix to each line
function read_and_process(filename, framePath) 
    local file = io.open(filename)
    local t = {}
    if file then
        for line in file:lines() do
            line = line:sub(1,#line-4) -- get rid of '.avi' to get the folder directory
            table.insert(t, framePath..'/'..line)
        end
    end
    return t
end

-- randomly choose (#batch) elements from (#total) elements
function getIndices(total, batch)
    --initialize a numbered list of size of total video number 
    local numLst = {}
    for i=1, total do
        numLst[i] = i
    end

    shuffleTable(numLst)
    local numTensor = torch.LongTensor(numLst)

    return numTensor[{{1,batch}}]
end

-- define the shuffle table function
function shuffleTable(t)
    local rand = math.random 
    assert(t, "shuffleTable() expected a table, got nil")    
    for i = #t, 2, -1 do
        local j = rand(i)
        t[i], t[j] = t[j], t[i]
    end
end
