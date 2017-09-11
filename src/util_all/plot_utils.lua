require "gnuplot"

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
