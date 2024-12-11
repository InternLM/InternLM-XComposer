interface Options {
    inputCodec: 'Int8' | 'Int16' | 'Int32' | 'Float32';
    channels: number;
    sampleRate: number;
    flushTime: number;
    onended?: (
        source: AudioBufferSourceNode,
        event: Event,
        duration: number,
        extraParams: Record<string, any>,
        isRefresh: boolean,
    ) => void;
    onstatechange?: (context: AudioContext, event: Event, state: AudioContextState) => void;
    extraParams?: Record<string, any>;
}

class PcmAudioPlayer {
    audioCtx?: AudioContext;
    gainNode?: GainNode;
    samples?: Float32Array;
    interval?: NodeJS.Timer;
    startTime = 0;
    extraParams?: Record<string, any>;
    currentAudioNode?: AudioBufferSourceNode | null;
    isStopped: boolean = false; // if audio player is stopped
    isPaused: boolean = false;

    option?: Options;
    convertValue?: number;
    typedArray!: Int8ArrayConstructor | Int16ArrayConstructor | Int32ArrayConstructor | Float32ArrayConstructor;

    constructor(option: Options) {
        this.init(option);
    }

    init(option: Options) {
        const defaultOption: Options = {
            inputCodec: option?.inputCodec,
            channels: option?.channels,
            sampleRate: option?.sampleRate,
            flushTime: option?.flushTime,
        };

        this.option = { ...defaultOption, ...option };
        this.samples = new Float32Array();
        this.interval = setInterval(this.flush.bind(this), this.option.flushTime);
        this.convertValue = this.getConvertValue();
        this.typedArray = this.getTypedArray();
        this.extraParams = {};
        this.initAudioContext();
        this.bindAudioContextEvent();
    }

    getConvertValue() {
        const inputCodecs = {
            Int8: 128,
            Int16: 32768,
            Int32: 2147483648,
            Float32: 1,
        };

        if (!inputCodecs[this.option!?.inputCodec]) {
            throw new Error('wrong codec.please input one of these codecs:Int8,Int16,Int32,Float32');
        }

        return inputCodecs[this.option!?.inputCodec];
    }

    getTypedArray() {
        const typedArrays = {
            Int8: Int8Array,
            Int16: Int16Array,
            Int32: Int32Array,
            Float32: Float32Array,
        };

        if (!typedArrays[this.option!?.inputCodec]) {
            throw new Error('wrong codec.please input one of these codecs:Int8,Int16,Int32,Float32');
        }

        return typedArrays[this.option!?.inputCodec];
    }

    initAudioContext() {
        this.audioCtx = new (window.AudioContext || (window as any)?.webkitAudioContext)();
        this.gainNode = this.audioCtx.createGain();
        this.gainNode.gain.value = 0.1;
        this.gainNode.connect(this.audioCtx.destination);
        this.startTime = this.audioCtx.currentTime;
    }

    isTypedArray(data: ArrayBufferView): data is ArrayBufferView {
        return (
            (data.byteLength && data.buffer && data.buffer.constructor == ArrayBuffer) ||
            data.constructor == ArrayBuffer
        );
    }

    isSupported(data: ArrayBufferView) {
        if (!this.isTypedArray(data)) {
            throw new Error('请传入ArrayBuffer或者任意TypedArray');
        }
        return true;
    }

    feed(data: ArrayBufferView) {
        this.isSupported(data);
        if (this.isStopped) {
            this.isStopped = false;
        }
        data = this.getFormattedValue(data);

        const tmp = new Float32Array(this?.samples!?.length + (data as any).length);
        tmp.set(this.samples as any, 0);
        tmp.set(data as any, this?.samples!.length);

        this.samples = tmp;
    }

    getFormattedValue(data: ArrayBuffer | ArrayBufferView): Float32Array {
        let newData;
        if (data.constructor == ArrayBuffer) {
            newData = new this.typedArray(data);
        } else {
            newData = new this.typedArray((data as ArrayBufferView)!.buffer);
        }

        const float32 = new Float32Array(newData.length);

        for (let i = 0; i < newData.length; i++) {
            float32[i] = newData[i] / this.convertValue!;
        }

        return float32;
    }

    volume(volume: number) {
        this.gainNode!.gain!.value = volume;
    }

    setExtraParams(data: Record<string, any>) {
        this.extraParams = data;
    }

    destroy() {
        if (this.interval) {
            clearInterval(this.interval as any);
        }

        this.samples = null as any;
        this.audioCtx!?.close();
        this.audioCtx = null as any;
    }

    flush() {
        if (!this.samples!?.length) return;

        const bufferSource = this.audioCtx!?.createBufferSource();

        const length = this.samples.length / this.option!?.channels;
        const allSampleLength = this.samples.length;
        const audioBuffer = this.audioCtx!?.createBuffer(this.option!?.channels, length, this.option!?.sampleRate);

        for (let channel = 0; channel < this.option!?.channels; channel++) {
            const audioData = audioBuffer.getChannelData(channel);
            let offset = channel;

            for (let i = 0; i < length; i++) {
                audioData[i] = this.samples[offset];

                if (i < 50) {
                    audioData[i] = (audioData[i] * i) / 50;
                }

                if (i >= length - 51) {
                    audioData[i] = (audioData[i] * (length - i - 1)) / 50;
                }

                offset += this?.option!?.channels as number;
            }
        }

        if ((this!.startTime as number) < this!?.audioCtx!?.currentTime) {
            this.startTime = this!?.audioCtx!?.currentTime;
        }
        bufferSource.buffer = audioBuffer;
        bufferSource.connect(this.gainNode!);
        bufferSource.start(this.startTime);
        this.currentAudioNode = bufferSource;

        const endTime = this.startTime;
        this.startTime += audioBuffer.duration;

        this.samples = new Float32Array();

        if (typeof this.option!.onended === 'function') {
            bufferSource.onended = (event) => {
                this.option!.onended!(bufferSource, event, audioBuffer.duration, this.extraParams!, this.isStopped);
            };
        }
    }

    async pause() {
        this.isPaused = true;
        await this.audioCtx!?.suspend();
    }

    async continue() {
        await this.audioCtx!?.resume();
    }

    async clearSample() {
        this.samples = new Float32Array();
    }

    async refresh() {
        if (this.currentAudioNode) {
            console.log('[pcm-audio-player] has currentAudioNode--------');
            // stop playing current audio
            await this.currentAudioNode.stop();
            // there is no audio is playing
            this.currentAudioNode = null;
            this.isStopped = true;
            this.startTime = 0;
        }

        // clear samples
        this.samples = new Float32Array();

        // to keep AudioContext alive
        await this?.audioCtx?.resume();
    }

    bindAudioContextEvent() {
        if (typeof this?.option!?.onstatechange === 'function') {
            this.audioCtx!.onstatechange = (event) => {
                if (this.audioCtx) {
                    this?.option!?.onstatechange!(this.audioCtx, event, this.audioCtx.state);
                }
            };
        }
    }
}

export default PcmAudioPlayer;
