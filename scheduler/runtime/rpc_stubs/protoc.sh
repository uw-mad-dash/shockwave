
ProtobufList=" common enums iterator_to_scheduler scheduler_to_worker worker_to_scheduler "

for protobuf in $ProtobufList; do
    python3 -m grpc_tools.protoc -I../protobuf --python_out=. \
    --grpc_python_out=. ../protobuf/$protobuf.proto
done