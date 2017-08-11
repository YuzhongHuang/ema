-- Yuzhong Huang yuzhong.huang@students.olin.edu
-- George Chen hc25@rice.edu

-- A data loading util stcript for HMDB-51. Given batch size, train\test data paths, 
-- splits txt files path, generates batch labels and batch inputs.

require 'cutorch'
require 'cunn'
require 'image'

math.randomseed(os.time())

-- getTest() return a table with a video tensor and a label tensor
function getTest(testpath, videoPath, frameNum, imgSize, channelNum, testBatchTotal, testName)
    local testset = {}
    local paths = {}
    local labels = {}

    paths, labels = getDataPath(testPath, videoPath, frameNum, imgSize, testBatchTotal, testName)

    testset.vids = getVideo(paths, frameNum, imgSize, channelNum)
    testset.labels = torch.Tensor(labels):cuda()

    return testset
end

-- getTrain() return a table with a video tensor and a label tensor for training
function getTrain(trainpath, videoPath, frameNum, imgSize, channelNum, trainBatchTotal, trainName)
    local trainset = {}
    local paths = {}
    local labels = {}

    paths, labels = getDataPath(trainPath, videoPath, frameNum, imgSize, trainBatchTotal, trainName)

    -- load whole video set
    trainset.vids = {}

    for i = 1, #paths do
        -- get current path
        local path = paths[i]

	if i%50 == 0 then
	    print("loaded videos : "..i)
	end

        -- deal with some formating issue with the file system
        local syspath = path:gsub('(%))', '\\%)')
        syspath = syspath:gsub('(%()', '\\%(')
        syspath = syspath:gsub('(%;)', '\\%;')
        syspath = syspath:gsub('(%&)', '\\%&')
        syspath = syspath:gsub('(%!)', '\\%!')
        syspath = syspath:gsub('(%])', '\\%]')
        syspath = syspath:gsub('(%[)', '\\%[')

        -- get all img files under the current folder
        local frames = ls(syspath)
        -- initialize current vid
        local vid = torch.FloatTensor(#frames, channelNum, imgSize, imgSize)

        -- load vid
        for j = 1, #frames do
            -- load a frame
            local frame = frames[j]
	    
            -- load and resize the image
            local img = image.load(path..'/'..frame, channelNum, 'float')
            img = image.scale(img, imgSize, imgSize)

            -- store the images into the tensor
            vid[{{j},{},{},{}}] = img
        end

        -- assign vid to trainset
        trainset.vids[i] = vid
    end

    -- load labels
    trainset.labels = torch.Tensor(labels)

    return trainset
end

function getDataPath(trainsets, videoPath, frameNum, imgSize, trainBatchTotal, fileName)
    local classes = ls(trainsets)
    local paths = {}
    local labels = {}
    local pathLabels = {}

    for i=1, #classes do
        local path = trainsets..'/'..classes[i]..fileName
        local framePath = videoPath..'/'..classes[i]
        local lst = read_and_process(path, framePath)
        
        -- get all shuffled elements per catergory for training
        local indices = getIndices(trainBatchTotal, trainBatchTotal) 
        
        -- get all trainning elements many videos from each class
        for j=1, trainBatchTotal do
            table.insert(pathLabels, lst[indices[j]]..' '..i)   -- combine paths and labels so for shuffling  
        end 
    end


    shuffleTable(pathLabels)    -- shuffle among classes

    -- separate paths and labels
    for i=1, #pathLabels do
        local path = split(pathLabels[i], " ")[1]
        local label = tonumber(split(pathLabels[i], " ")[2])
        table.insert(paths, path)
        table.insert(labels, label)
    end

    return paths, labels
end

function getVideo(paths, frameNum, imgSize, channelNum, data_aug)
    local batchInputs = torch.FloatTensor(#paths, frameNum, channelNum, imgSize, imgSize)
    for i=1, #paths do
        local path = paths[i]
	    if i%50 == 0 then
	        print("loaded videos : "..i)
	    end


        -- deal with some formating issue with the file system
        local syspath = path:gsub('(%))', '\\%)')
        syspath = syspath:gsub('(%()', '\\%(')
        syspath = syspath:gsub('(%;)', '\\%;')
        syspath = syspath:gsub('(%&)', '\\%&')
	syspath = syspath:gsub('(%!)', '\\%!')
	syspath = syspath:gsub('(%])', '\\%]')    
        syspath = syspath:gsub('(%[)', '\\%[')
        -- get all img files under the current folder
        local frames = ls(syspath)

        -- data augmentation
        start = 1
        if data_aug then
            if frameNum > #frames then
                start = math.random(frameNum - #frames)
            end
        end

        -- iterate through min(frameNum, #frames) times
        for j=start, math.min(frameNum, #frames) do
            -- load a frame
            local frame = frames[j]

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

-- define a string split function
function split(inputstr, sep)
        if sep == nil then
                sep = "%s"
        end
        local t={} ; i=1
        for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
                t[i] = str
                i = i + 1
        end
        return t
end
