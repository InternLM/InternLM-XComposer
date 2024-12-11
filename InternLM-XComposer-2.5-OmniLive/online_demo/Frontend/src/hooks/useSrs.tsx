import { ChatContext } from '@/context/chat-context';
import { useContext, useEffect, useRef, useState } from 'react';
import { CameraStatus } from '@/components/camera-video';
// @ts-ignore
import Srs from '@/components/srs/srs.sdk.js';
import { SRS_BASE_URL } from '@/config/service-url';

const useSrs = () => {
    const { sessionId } = useContext(ChatContext);
    const ref = useRef<any>();
    const sdkRef = useRef<any>();
    const srsReadyRef = useRef<boolean>(false);
    const [cameraStatus, setCameraStatus] = useState<CameraStatus>(CameraStatus.DEFAULT);

    const setCameraVideo = (stream: MediaStream | null) => {
        if (stream) {
            setCameraStatus(CameraStatus.AVAILABLE);
            const video = ref.current as HTMLVideoElement;
            video.srcObject = stream;
        } else {
            setCameraStatus(CameraStatus.ERROR);
        }
    };

    const initSrs = (callback: any) => {
        if (sdkRef.current) {
            sdkRef.current.close();
        }
        console.log('[useSrs] initSrs', sessionId);
        sdkRef.current = new Srs.SrsRtcPublisherAsync();
        const url = `${SRS_BASE_URL}${sessionId}`;
        setTimeout(() => {
            sdkRef.current
                .publish(url, setCameraVideo)
                .then((session: any) => {
                    console.log('[useSrs]publish suc', session);
                    callback();
                })
                .catch((reason: any) => {
                    console.error('[useSrs]publish fail', reason);
                    if (reason instanceof DOMException) {
                        setCameraStatus(CameraStatus.ERROR);
                    }
                    sdkRef.current.close();
                });
        }, 200);
    };

    useEffect(() => {
        return () => {
            sdkRef.current && sdkRef.current.close();
            sdkRef.current = null;
        };
    }, [sessionId]);

    return {
        ref,
        srsReadyRef,
        cameraStatus,
        initSrs,
    };
};

export default useSrs;
