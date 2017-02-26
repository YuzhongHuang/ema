require "gnuplot"

-- a function that plots the overall accuracy
function plot(name, iter, a)
	gnuplot.pngfigure(name..".png")
	gnuplot.plot(torch.Tensor(iter), torch.Tensor(a))

	gnuplot.title('LRCN Training Plot')
	gnuplot.xlabel('Iterations (epoch)')
	gnuplot.ylabel('Accuracy (percent)')

	gnuplot.plotflush()
end