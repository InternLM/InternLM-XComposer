export const calculatePCMAudioDuration = (arrayBuffer: any, sampleRate: any, bitDepth: any, channels: any) => {
    const bytesPerSample = bitDepth / 8;
    const view = new DataView(arrayBuffer);
    const totalSamples = arrayBuffer.byteLength / (bytesPerSample * channels);
    const durationInSeconds = totalSamples / sampleRate;

    return durationInSeconds;
};
