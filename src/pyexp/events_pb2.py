# -*- coding: utf-8 -*-
"""Protocol buffer message definitions for pyexp event logging.

Uses runtime descriptor construction to avoid needing protoc compilation.
"""
from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import message_factory


def _create_messages():
    """Create protobuf message classes at runtime."""
    pool = descriptor_pool.DescriptorPool()

    # Create file descriptor
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "events.proto"
    file_proto.package = "pyexp"
    file_proto.syntax = "proto3"

    # Scalar message
    scalar_msg = file_proto.message_type.add()
    scalar_msg.name = "Scalar"
    field = scalar_msg.field.add()
    field.name = "tag"
    field.number = 1
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field = scalar_msg.field.add()
    field.name = "value"
    field.number = 2
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL

    # Text message
    text_msg = file_proto.message_type.add()
    text_msg.name = "Text"
    field = text_msg.field.add()
    field.name = "tag"
    field.number = 1
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field = text_msg.field.add()
    field.name = "value"
    field.number = 2
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL

    # Figure message
    figure_msg = file_proto.message_type.add()
    figure_msg.name = "Figure"
    field = figure_msg.field.add()
    field.name = "tag"
    field.number = 1
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field = figure_msg.field.add()
    field.name = "data"
    field.number = 2
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_BYTES
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field = figure_msg.field.add()
    field.name = "interactive"
    field.number = 3
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_BOOL
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL

    # Checkpoint message
    checkpoint_msg = file_proto.message_type.add()
    checkpoint_msg.name = "Checkpoint"
    field = checkpoint_msg.field.add()
    field.name = "tag"
    field.number = 1
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field = checkpoint_msg.field.add()
    field.name = "data"
    field.number = 2
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_BYTES
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL

    # Event message (contains oneof)
    event_msg = file_proto.message_type.add()
    event_msg.name = "Event"
    field = event_msg.field.add()
    field.name = "timestamp"
    field.number = 1
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field = event_msg.field.add()
    field.name = "iteration"
    field.number = 2
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT64
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    # oneof fields
    field = event_msg.field.add()
    field.name = "scalar"
    field.number = 3
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".pyexp.Scalar"
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.oneof_index = 0
    field = event_msg.field.add()
    field.name = "text"
    field.number = 4
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".pyexp.Text"
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.oneof_index = 0
    field = event_msg.field.add()
    field.name = "figure"
    field.number = 5
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".pyexp.Figure"
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.oneof_index = 0
    field = event_msg.field.add()
    field.name = "checkpoint"
    field.number = 6
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".pyexp.Checkpoint"
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.oneof_index = 0
    # Define the oneof
    oneof = event_msg.oneof_decl.add()
    oneof.name = "data"

    # Add to pool
    pool.Add(file_proto)

    # Create message classes using modern API (protobuf 4.x compatible)
    return {
        "Event": message_factory.GetMessageClass(
            pool.FindMessageTypeByName("pyexp.Event")
        ),
        "Scalar": message_factory.GetMessageClass(
            pool.FindMessageTypeByName("pyexp.Scalar")
        ),
        "Text": message_factory.GetMessageClass(
            pool.FindMessageTypeByName("pyexp.Text")
        ),
        "Figure": message_factory.GetMessageClass(
            pool.FindMessageTypeByName("pyexp.Figure")
        ),
        "Checkpoint": message_factory.GetMessageClass(
            pool.FindMessageTypeByName("pyexp.Checkpoint")
        ),
    }


_messages = _create_messages()

Event = _messages["Event"]
Scalar = _messages["Scalar"]
Text = _messages["Text"]
Figure = _messages["Figure"]
Checkpoint = _messages["Checkpoint"]
