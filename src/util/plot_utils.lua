require "gnuplot"

-- a function that plots the overall accuracy
function plot(name, iter, a, e)
	gnuplot.pngfigure(name..".png")
	
	gnuplot.plot(
		{'Training', torch.Tensor(iter), torch.Tensor(e), '~'},
		{"Testing", torch.Tensor(iter), torch.Tensor(a), '~'}
	)

	gnuplot.title('LRCN Training Plot')
	gnuplot.xlabel('Iterations (epoch)')
	gnuplot.ylabel('Accuracy (percent)')

	gnuplot.plotflush()
end