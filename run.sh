#!/bin/bash

echo "Compile curveFitting..."
make "SMS=75"

if [ -f curveFitting ]
then
	case $# in
		2)
			file=$1
			nSamples=$(cat $1 | wc -l)
			order=$2
			
			./curveFitting $file $nSamples $order
			;;
			
		0)
			./curveFitting
			;;
		*)
			echo "Wrong argument count"
	esac
else
	echo "Compile the program with make and specified number of SMS"
fi
