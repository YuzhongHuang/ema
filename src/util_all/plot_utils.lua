require "gnuplot"
require "lfs"
require "image"

-- a function that plots the overall accuracy
function plot(name, iter, a, e)
	gnuplot.pngfigure(name..".png")
	
	gnuplot.plot(
		{'Loss', torch.Tensor(iter), torch.Tensor(e), '~'},
		{"Accuracy", torch.Tensor(iter), torch.Tensor(a), '~'}
	)

	gnuplot.title('Training Loss and Testing Accuracy')
	gnuplot.xlabel('Epochs')
	gnuplot.ylabel('Accuracy and Loss')

	gnuplot.plotflush()
end

-- get_weight() takes a network with a given indices to a layer, 
-- and a video input. Generate folders of images outputs of that layer's 
-- weights and activation to given location with a given filename

-- vid should be in the form of (1, frameNum, channelNum, imgSize, imgSize)
-- model will need
function get_weight(indices, model, vid, filename, save_path)
	-- potential bug: need to call Data Paralell

	-- forward pass the video to the network
	model:forward(vid:cuda())  -- :cuda() needed for gpu-trained networks

	-- get to the layer
	local layer = model
	for i = 1, #indices do
		layer = model.modules[indices[i]]
	end

	-- get the weight and activation
	local weights = layer.weight
	local outputs = layer.output

	-- save the activation images
	for i = 1, (#outputs)[2] do -- loop through channels
	    vid_path = save_path.."/"..filename.."_channel"..i
	    lfs.mkdir(vid_path)
	    for j = 1, (#outputs)[1] do
		image.save(vid_path.."img_naming"..".png", outputs[j][i])	
	    end
	end
end
