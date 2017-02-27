-- Yuzhong Huang yuzhong.huang@students.olin.edu
-- George Chen hc25@rice.edu

-- A data loading util stcript for HMDB-51. Given batch size, train\test data paths, 
-- splits txt files path, generates batch labels and batch inputs.

require 'cutorch'
require 'cunn'
require 'image'

math.randomseed(os.time())

-- getTest(path) return a table with a video tensor and a label tensor
function getTest(testpath, videoPath, frameNum, batchSize, imgSize)
    local classes = ls(testPath)
    local videoPaths = {}
    local testLabels = {}
    local t = {}

    for i=1, #classes do
        local path = testPath..'/'..classes[i]..'/test.txt'
        local framePath = videoPath..'/'..classes[i]
        local lst = read_and_process(path, framePath)
        local indices = getIndices(30, batchSize)

        for j=1, batchSize do
            table.insert(videoPaths, lst[indices[j]])
            table.insert(testLabels, i)
        end
    end

    t.vids = getVideo(videoPaths, frameNum, imgSize)
    t.labels = torch.Tensor(testLabels)

    return t
end

function getBatch(trainsets, videoPath, batchSize, frameNum, imgSize)
    local classes = ls(trainsets)
    local batchPaths = {}
    local batchLabels = {}

    for i=1, #classes do
        local path = trainsets..'/'..classes[i]..'/train.txt'
        local framePath = videoPath..'/'..classes[i]
        local lst = read_and_process(path, framePath)
        local indices = getIndices(70, batchSize) -- 70 elements per catergory for training

        -- get #batchSize many videos from each class
        for j=1, batchSize do
            table.insert(batchPaths, lst[indices[j]])
            table.insert(batchLabels, i)
        end
    end

    return getVideo(batchPaths, frameNum, imgSize), torch.Tensor(batchLabels):cuda()
end

function getEpoch(trainsets, videoPath, frameNum, imgSize)
    local classes = ls(trainsets)
    local Paths = {}
    local Labels = {}

    for i=1, #classes do
        local path = trainsets..'/'..classes[i]..'/train.txt'
        local framePath = videoPath..'/'..classes[i]
        local lst = read_and_process(path, framePath)
        local indices = getIndices(70, 70) -- get all 70 shuffled elements per catergory for training

        -- get all 70 trainning elements many videos from each class
        for j=1, batchSize do
            table.insert(Paths, lst[indices[j]])
            table.insert(Labels, i)
        end 
    end

    return batchPaths, torch.Tensor(batchLabels):cuda()
end

function getVideo(paths, frameNum, imgSize)
    -- local max = maxSequence(paths)
    local batchInputs = torch.FloatTensor(#paths, frameNum, 3, imgSize, imgSize)
    for i=1, #paths do
        local path = paths[i]

        -- deal with some formating issue with the file system
        sys_path = path:gsub('(%))', '\\%)')
        sys_path = path:gsub('(%()', '\\%(')
        sys_path = path:gsub('(%;)', '\\%;')
        sys_path = path:gsub('(&)', '\\&')
        local frames = ls(sys_path)
        local num = #frames

        if num > frameNum then
            num = frameNum
        end

        for j=1, num do
            -- read the image
            local frame = frames[j]
            local img = image.load(path..'/'..frame)
            img = image.scale(img, imgSize, imgSize)

            -- store the images with a rescaled version
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
