import 'dart:ffi';// For FFI
import 'dart:io'; // For Platform.isX
import 'package:ffi/ffi.dart';
import 'dart:isolate';

// C function signatures
typedef _version_func = Pointer<Utf8> Function();
typedef _process_image_func = Int32 Function(Pointer<Utf8>);

// Dart function signatures
typedef _VersionFunc = Pointer<Utf8> Function();
typedef _ProcessImageFunc = int Function(Pointer<Utf8>);

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

int processImage(ProcessImageArguments args) {
    var result = _processImage(Utf8.toUtf8(args.inputPath));
    args.port.send(result);
    return result;
}

class ProcessImageArguments {
    final String inputPath;
    final SendPort port;
    ProcessImageArguments(this.inputPath, this.port);
}

/*

sourceSets {
        main.java.srcDirs += 'src/main/kotlin'
        main {
            jniLibs.srcDirs = [ 'libs','src/main/nativeLibs']  // libs
        }
    }
    如果报*.so文件重复,注释sourceSets里面的main {
            jniLibs.srcDirs = [ 'libs','src/main/nativeLibs']  // libs
        }
 */