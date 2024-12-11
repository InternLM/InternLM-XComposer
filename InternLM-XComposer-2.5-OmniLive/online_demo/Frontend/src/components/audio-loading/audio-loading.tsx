import classNames from 'classnames';
import style from './audio-loading.module.less';

interface IAudioLoadingProps {
    height?: number; // loading box height
    gap?: number; // loading section gap
    forbidAnimate?: boolean; // if forbid animation
    count?: number; // loading section count
}

const AudioLoading = (props: IAudioLoadingProps) => {
    const { height = 32, forbidAnimate = false, gap = 8, count = 5 } = props;

    return (
        <div
            className={classNames(style['audio-loading'], forbidAnimate && style['forbid-animate'])}
            style={{ height }}
        >
            {new Array(count).fill(1).map((item: number, index: number) => (
                <div key={index} style={{ height, margin: `0 ${gap}px` }}></div>
            ))}
        </div>
    );
};

export default AudioLoading;
