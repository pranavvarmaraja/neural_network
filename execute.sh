if [ -d "./bin" ] 
then
    echo "Directory ./bin already exists." 
else
    mkdir bin
fi
javac -cp ./lib/json-simple-1.1.1.jar -s ./bin/Network ./src/Network.java -Xlint:non
mv ./src/*.class ./bin
java -cp ./bin:./lib/json-simple-1.1.1.jar Network $1
