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
final _ProcessImageFunc _processSemiQuanImage = nativeAddLib
    .lookup<NativeFunction<_process_image_func>>('process_semi_quan_image')
    .asFunction();

String opencvVersion() {
    return _version().toDartString();
}

int processImage(ProcessImageArguments args) {
    if (args.type == 0) {
        var result = _processImage(args.inputPath.toNativeUtf8());
        return result;
    } else {
        var result = _processSemiQuanImage(args.inputPath.toNativeUtf8());
        return result;
    }
}

class ProcessImageArguments {
    final String inputPath;
    final int type;
    ProcessImageArguments(this.inputPath, this.type);
}

/*
iOS 需要修改一下配置
When creating a release archive (IPA) the symbols are stripped by Xcode.
1. In Xcode, go to **Target Runner > Build Settings > Strip Style**.
2. Change from **All Symbols** to **Non-Global Symbols**.

iOS 需要拷贝一份native_add.cpp到iOS独立项目里面

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