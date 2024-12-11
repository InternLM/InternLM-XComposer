import { ChatContext, IChatStatus } from '@/context/chat-context';
import { useContext, useEffect, useRef } from 'react';
import PcmAudioPlayer from '@/components/pcm-player/pcm-audio-player';
import { calculatePCMAudioDuration } from '@/components/pcm-player/pcm-player-util';
import { BIT_DEPTH, CHANNELS, FLUSH_TIME, INPUT_CODE_C, SAMPLE_RATE } from '@/components/pcm-player/pcm-config';

const usePlayer = () => {
    const playerRef = useRef<any>();
    const audioLength = useRef(0); // audio playing length
    const outpuingLength = useRef(0); // model outputing length
    const startFlag = useRef(false); // if voice starts
    const { setStatus } = useContext(ChatContext);

    const handleBuffer = async (streamBuffer: any) => {
        playerRef.current && playerRef.current?.feed(streamBuffer);
    };

    const closeAudio = async () => {
        await playerRef.current?.destroy();
        playerRef.current = null;
        outpuingLength.current = 0;
        audioLength.current = 0;
        startFlag.current = false;
        setStatus(IChatStatus.LISTENING);
    };

    const createPlayer = () => {
        console.log('[usePlayer] create new audio player------');
        playerRef.current = new PcmAudioPlayer({
            inputCodec: INPUT_CODE_C,
            channels: CHANNELS,
            sampleRate: SAMPLE_RATE,
            flushTime: FLUSH_TIME,
            onstatechange: () => {
                return {};
            },
            onended: (node, event, duration, extraParams, isrefresh) => {
                console.info('[usePlayer] onended', node, event, duration, extraParams, isrefresh);
                if (!isrefresh) {
                    audioLength!.current += duration;
                }
                console.log('[usePlayer] onended audioLength', audioLength);
                // if the audio is finishing playing by comparing the model outputing length and the audio playing length
                if (Math?.abs(outpuingLength.current - audioLength?.current) < 1) {
                    console.info('[usePlayer] onend finally');
                    !isrefresh && console.info('[usePlayer] close audio - play finished');
                    !isrefresh && closeAudio();
                }

                return {};
            },
        });
    };

    const readBuffer = (data: any) => {
        const reader = new FileReader();
        reader.onload = function (e: any) {
            const arrayBuffer = e.target.result;
            const uint8Array = new Uint8Array(arrayBuffer);
            const buffer = uint8Array.buffer;
            handleBuffer(buffer);
            outpuingLength.current += calculatePCMAudioDuration(buffer, SAMPLE_RATE, BIT_DEPTH, CHANNELS);
            console.log('[usePlayer] pcm reader', outpuingLength);
        };
        reader.readAsArrayBuffer(data);
    };

    useEffect(() => {
        if (!playerRef.current) {
            createPlayer();
        }
    }, []);

    return {
        playerRef,
        startFlag,
        closeAudio,
        readBuffer,
        createPlayer,
        handleBuffer,
    };
};

export default usePlayer;
