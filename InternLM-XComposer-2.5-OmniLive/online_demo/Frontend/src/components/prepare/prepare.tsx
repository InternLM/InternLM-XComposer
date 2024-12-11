import { useCallback, useContext, useEffect, useMemo, useRef } from 'react';
import style from './prepare.module.less';
import { ChatContext, IChatStatus } from '@/context/chat-context';
import lottie from 'lottie-web';

const PREPARE_ANIMATION_CONTAINER_ID = 'PREPARE_ANIMATION_CONTAINER';

const Prepare = () => {
    const { status } = useContext(ChatContext);
    const animationReadyRef = useRef<boolean>(false);

    const statusText = useMemo(() => {
        switch (status) {
            case IChatStatus.PREPARING:
                return '准备中';
            case IChatStatus.LISTENING:
                return '我在听';
            case IChatStatus.FORCE_CLOSE:
            case IChatStatus.SOCKET_ERROR:
                return '重连中';
            default:
                return '';
        }
    }, [status]);

    const initAnimation = useCallback(() => {
        const ele = document.getElementById(PREPARE_ANIMATION_CONTAINER_ID);
        if (ele) {
            lottie.loadAnimation({
                container: ele,
                renderer: 'svg',
                loop: true,
                autoplay: true,
                path: `/src/assets/json/animation.json`,
            });
        }
    }, []);

    useEffect(() => {
        if (!animationReadyRef.current) {
            initAnimation();
            animationReadyRef.current = true;
        }

        return () => {
            animationReadyRef.current = false;
        };
    }, []);

    return (
        <div className={style.prepare}>
            <div className={style.prepareAnimation} id={PREPARE_ANIMATION_CONTAINER_ID}></div>
            <div className={style.prepareText}>{statusText}</div>
        </div>
    );
};

export default Prepare;
