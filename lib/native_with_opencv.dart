import 'dart:ffi';// For FFI
import 'dart:io'; // For Platform.isX
import 'package:ffi/ffi.dart';

// C function signatures
typedef _version_func = Pointer<Utf8> Function();
typedef _process_image_func = Void Function(Pointer<Utf8>, Pointer<Utf8>);

// Dart function signatures
typedef _VersionFunc = Pointer<Utf8> Function();
typedef _ProcessImageFunc = void Function(Pointer<Utf8>, Pointer<Utf8>);

final DynamicLibrary nativeAddLib =
Platform.isAndroid ? DynamicLibrary.open("libnative_with_opencv.so") : DynamicLibrary.process();

final int Function(int x, int y) nativeAdd =
nativeAddLib.lookup<NativeFunction<Int32 Function(Int32, Int32)>>("native_add").asFunction();

// Looking for the functions
final _VersionFunc _version = nativeAddLib
    .lookup<NativeFunction<_version_func>>('version').asFunction();
final _ProcessImageFunc _processImage = nativeAddLib
    .lookup<NativeFunction<_process_image_func>>('process_image')
    .asFunction();

String opencvVersion() {
    return Utf8.fromUtf8(_version());
}

void processImage(ProcessImageArguments args) {
    _processImage(Utf8.toUtf8(args.inputPath), Utf8.toUtf8(args.outputPath));
}

class ProcessImageArguments {
    final String inputPath;
    final String outputPath;

    ProcessImageArguments(this.inputPath, this.outputPath);
}