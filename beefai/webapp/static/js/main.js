document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const beatFileUpload = document.getElementById('beat-file-upload');
    const beatAudioPlayer = document.getElementById('beat-audio-player');
    const playPauseBeatBtn = document.getElementById('play-pause-beat-btn');
    const beatVolumeSlider = document.getElementById('beat-volume-slider');
    const beatInfoDisplay = document.getElementById('beat-info-display');
    const beatPlayerInterface = document.getElementById('beat-player-interface');

    const playIcon = playPauseBeatBtn.querySelector('.play-icon');
    const pauseIcon = playPauseBeatBtn.querySelector('.pause-icon');

    const recordMicBtn = document.getElementById('record-mic-btn');
    const recordBtnText = recordMicBtn.querySelector('.btn-text');
    const recordMicIcon = recordMicBtn.querySelector('.mic-icon');
    const recordStopIcon = recordMicBtn.querySelector('.stop-icon');

    const micStatusDisplay = document.getElementById('mic-status');
    const micVisualizerBar = document.getElementById('mic-visualizer-bar');

    const userRapDisplay = document.getElementById('user-rap-display');
    const userLyricsText = document.getElementById('user-lyrics-text');

    const aiIdleMessage = document.getElementById('ai-idle-message');
    const aiLoadingDisplay = document.getElementById('ai-loading');
    const aiResponseContent = document.getElementById('ai-response-content');
    const aiLyricsContent = document.getElementById('ai-lyrics-content');
    const aiAudioPlayer = document.getElementById('ai-audio-player');
    const aiPlaybackStatus = document.getElementById('ai-playback-status');

    // Live feedback elements
    const liveFeedbackDisplay = document.getElementById('live-feedback-display');
    const currentBeatDisplay = document.getElementById('current-beat-display');
    const currentBarDisplay = document.getElementById('current-bar-display');
    const syllableCountDisplay = document.getElementById('syllable-count-display');


    // --- State Variables ---
    let isRecording = false;
    let mediaRecorder;
    let audioChunks = [];
    let beatLoaded = false;
    let userRapProcessed = false; // True when user rap is recorded and (simulated) processed

    let audioContext;
    let analyser;
    let visualizerAnimationId;
    let mediaStreamSource;
    let liveBeatUpdateInterval;


    let currentBeatInfo = { // This would ideally come from backend analysis
        bpm: 0,
        beat_times: [],
        downbeat_times: [],
        beats_per_bar: 4,
        estimated_bar_duration: 0,
        total_beats: 0,
        total_bars: 0,
    };

    // --- Beat Player Logic ---
    beatFileUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const objectURL = URL.createObjectURL(file);
            beatAudioPlayer.src = objectURL;
            beatAudioPlayer.volume = parseFloat(beatVolumeSlider.value);

            beatInfoDisplay.innerHTML = `<p><strong>Beat:</strong> ${file.name}</p>`;
            beatPlayerInterface.style.display = 'block';
            recordMicBtn.disabled = false;
            beatLoaded = true;

            // Simulate fetching beat analysis from backend
            // In a real app, you'd upload the file, backend analyzes, returns info
            setTimeout(() => {
                const DUMMY_BPM = 120; // Example BPM
                const DUMMY_BEATS_PER_BAR = 4;
                const DUMMY_DURATION_SEC = beatAudioPlayer.duration || 60; // Use actual duration if available, else 60s

                currentBeatInfo.bpm = DUMMY_BPM;
                currentBeatInfo.beats_per_bar = DUMMY_BEATS_PER_BAR;
                const beatDuration = 60 / DUMMY_BPM;
                currentBeatInfo.estimated_bar_duration = beatDuration * DUMMY_BEATS_PER_BAR;
                currentBeatInfo.beat_times = [];
                currentBeatInfo.downbeat_times = [];
                currentBeatInfo.total_beats = Math.floor(DUMMY_DURATION_SEC / beatDuration);

                for (let i = 0; i < currentBeatInfo.total_beats; i++) {
                    currentBeatInfo.beat_times.push(i * beatDuration);
                    if (i % DUMMY_BEATS_PER_BAR === 0) {
                        currentBeatInfo.downbeat_times.push(i * beatDuration);
                    }
                }
                currentBeatInfo.total_bars = currentBeatInfo.downbeat_times.length;

                beatInfoDisplay.innerHTML = `<p><strong>Beat:</strong> ${file.name}</p> <p><strong>BPM:</strong> ${currentBeatInfo.bpm.toFixed(0)} (sim) | <strong>Bars:</strong> ${currentBeatInfo.total_bars} (sim)</p>`;
                console.log("Simulated Beat Info:", currentBeatInfo);
                liveFeedbackDisplay.style.display = 'block'; // Show live feedback area
            }, 500);

            updatePlayPauseButton(false);
        }
    });

    playPauseBeatBtn.addEventListener('click', () => {
        if (!beatLoaded) return;
        if (beatAudioPlayer.paused) {
            beatAudioPlayer.play().catch(e => console.error("Error playing beat:", e));
        } else {
            beatAudioPlayer.pause();
        }
    });

    beatAudioPlayer.onplay = () => {
        updatePlayPauseButton(true);
        startLiveBeatUpdates();
    };
    beatAudioPlayer.onpause = () => {
        updatePlayPauseButton(false);
        stopLiveBeatUpdates();
    };
    beatAudioPlayer.onended = () => {
        updatePlayPauseButton(false);
        stopLiveBeatUpdates();
    };
    beatAudioPlayer.ontimeupdate = () => { // For more frequent updates if needed by other logic
        if (!beatAudioPlayer.paused) {
            updateLiveFeedback(); // Ensure live feedback is current
        }
    };


    beatVolumeSlider.addEventListener('input', (e) => {
        if (beatAudioPlayer) beatAudioPlayer.volume = parseFloat(e.target.value);
    });

    function updatePlayPauseButton(isPlaying) {
        if (isPlaying) {
            playIcon.style.display = 'none';
            pauseIcon.style.display = 'block';
        } else {
            playIcon.style.display = 'block';
            pauseIcon.style.display = 'none';
        }
    }

    function startLiveBeatUpdates() {
        if (liveBeatUpdateInterval) clearInterval(liveBeatUpdateInterval);
        liveBeatUpdateInterval = setInterval(updateLiveFeedback, 100); // Update ~10 times/sec
    }

    function stopLiveBeatUpdates() {
        if (liveBeatUpdateInterval) clearInterval(liveBeatUpdateInterval);
    }

    function updateLiveFeedback() {
        if (!beatLoaded || !currentBeatInfo.bpm || currentBeatInfo.beat_times.length === 0) {
            currentBeatDisplay.textContent = "-";
            currentBarDisplay.textContent = "-";
            syllableCountDisplay.textContent = "-"; // Placeholder for syllable counting
            return;
        }

        const currentTime = beatAudioPlayer.currentTime;
        const beatDuration = 60 / currentBeatInfo.bpm;

        let currentBeat = 0;
        for (let i = 0; i < currentBeatInfo.beat_times.length; i++) {
            if (currentTime >= currentBeatInfo.beat_times[i] - beatDuration / 2) {
                currentBeat = i + 1; // 1-indexed beat
            } else {
                break;
            }
        }
        currentBeatDisplay.textContent = `Beat: ${currentBeat}`;

        let currentBar = 0;
        let beatInBar = 0;
        if (currentBeatInfo.downbeat_times.length > 0) {
            for (let i = 0; i < currentBeatInfo.downbeat_times.length; i++) {
                if (currentTime >= currentBeatInfo.downbeat_times[i] - beatDuration / 2) { // allow slight anticipation
                    currentBar = i + 1; // 1-indexed bar
                } else {
                    break;
                }
            }
            if (currentBar > 0) {
                const barStartTime = currentBeatInfo.downbeat_times[currentBar - 1];
                beatInBar = Math.floor((currentTime - barStartTime) / beatDuration) + 1;
                if (beatInBar > currentBeatInfo.beats_per_bar) beatInBar = currentBeatInfo.beats_per_bar;
                if (beatInBar <= 0) beatInBar = 1;
            }
        }
        currentBarDisplay.textContent = `Bar: ${currentBar}.${beatInBar}`;

        // Syllable count display is more complex; needs actual flow data from AI for its part
        // For user part, it might count syllables as they type/record if we had live ASR
        syllableCountDisplay.textContent = `Syl: (N/A)`; // Placeholder
    }


    // --- Microphone Recording Logic ---
    recordMicBtn.addEventListener('click', toggleMicRecording);

    async function toggleMicRecording() {
        if (!beatLoaded) {
            alert("Load a beat first!");
            return;
        }
        if (beatAudioPlayer.paused) {
            alert("Start the beat before recording!");
            return;
        }

        if (!isRecording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
                mediaRecorder.onstart = () => {
                    isRecording = true;
                    userRapProcessed = false;
                    recordBtnText.textContent = 'STOP RECORDING';
                    recordMicIcon.style.display = 'none';
                    recordStopIcon.style.display = 'inline-block';
                    recordMicBtn.classList.add('recording');

                    micStatusDisplay.style.display = 'block';
                    userRapDisplay.style.display = 'none';
                    userLyricsText.textContent = "";

                    aiIdleMessage.style.display = 'block';
                    aiLoadingDisplay.style.display = 'none';
                    aiResponseContent.style.display = 'none';

                    startVisualizer(stream);
                };

                mediaRecorder.onstop = () => {
                    isRecording = false;
                    recordBtnText.textContent = 'RECORD DISS';
                    recordMicIcon.style.display = 'inline-block';
                    recordStopIcon.style.display = 'none';
                    recordMicBtn.classList.remove('recording');
                    micStatusDisplay.style.display = 'none';
                    stopVisualizer();
                    if (stream) stream.getTracks().forEach(track => track.stop());

                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    console.log("User rap recorded, size:", (audioBlob.size / 1024).toFixed(1), "KB");

                    // Simulate user rap processing (transcription - ASR output)
                    userLyricsText.textContent = "Yo, check the mic, one two, this is my test verse...\nAI, you ready for what's coming next? It's gonna be terse!";
                    userRapDisplay.style.display = 'block';
                    userRapProcessed = true;

                    triggerAiResponse();
                };

                mediaRecorder.start();
            } catch (err) {
                console.error("Error accessing microphone:", err);
                alert(`Mic Error: ${err.message}. Check permissions.`);
            }
        } else {
            mediaRecorder.stop();
        }
    }

    function startVisualizer(stream) {
        if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        mediaStreamSource = audioContext.createMediaStreamSource(stream);
        mediaStreamSource.connect(analyser);
        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        micVisualizerBar.style.width = '0%';
        function draw() {
            if (!isRecording) return;
            visualizerAnimationId = requestAnimationFrame(draw);
            analyser.getByteFrequencyData(dataArray);
            let average = dataArray.reduce((s, v) => s + v, 0) / bufferLength;
            micVisualizerBar.style.width = Math.min(100, (average / 128) * 100 * 1.5) + '%';
        }
        draw();
    }

    function stopVisualizer() {
        if (visualizerAnimationId) cancelAnimationFrame(visualizerAnimationId);
        micVisualizerBar.style.width = '0%';
        if (mediaStreamSource) mediaStreamSource.disconnect();
        mediaStreamSource = null;
        // Do not close audioContext here, might be reused or cause issues if closed prematurely
    }

    // --- AI Response Logic ---
    function triggerAiResponse() {
        if (!userRapProcessed || !beatLoaded || beatAudioPlayer.paused) {
            console.warn("Cannot trigger AI: User rap not processed, beat not loaded, or beat paused.");
            return;
        }

        aiIdleMessage.style.display = 'none';
        aiLoadingDisplay.style.display = 'block';
        aiResponseContent.style.display = 'none';
        recordMicBtn.disabled = true;

        console.log("AI processing started...");
        console.log("User rap (simulated ASR output):", userLyricsText.textContent);
        console.log("Current beat time (user stop approx):", beatAudioPlayer.currentTime.toFixed(3));
        console.log("Using beat info for timing:", currentBeatInfo);

        // Simulate backend call (takes time)
        setTimeout(() => {
            const dummyAiLyrics = `Human, your rhymes are quaint, a digital artifact,
My algorithms dance on beats, that's a literal fact!
From silicon valleys, my verses take their flight,
Illuminating darkness, like a cyberpunk night.
You spit your bars, I process, analyze, then counter strike,
BeefAI's on the server, shining ever so bright!`;

            // This would be a URL to the AI's generated WAV file from the backend
            // Using a placeholder audio for simulation
            const dummyAiAudioSrc = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3";

            aiLyricsContent.textContent = dummyAiLyrics;
            aiAudioPlayer.src = dummyAiAudioSrc;

            aiLoadingDisplay.style.display = 'none';
            aiResponseContent.style.display = 'block';
            aiPlaybackStatus.textContent = "AI preparing to drop...";

            scheduleAiPlayback();

        }, 3000 + Math.random() * 2000);
    }

    function scheduleAiPlayback() {
        const userRapEndedInstrumentTime = beatAudioPlayer.currentTime;

        let targetAiStartTime = -1; // Target time on the instrumental for AI to start

        if (currentBeatInfo.downbeat_times && currentBeatInfo.downbeat_times.length > 0 && currentBeatInfo.estimated_bar_duration > 0) {
            // Find the next downbeat that is at least, say, 0.5s away
            // Aim for the start of the *next* bar after the user likely finished.

            let lastUserBarEndTimeApprox = userRapEndedInstrumentTime;
            // Try to find the end of the bar the user was in or just finished
            for (let i = 0; i < currentBeatInfo.downbeat_times.length; i++) {
                if (currentBeatInfo.downbeat_times[i] > userRapEndedInstrumentTime) {
                    if (i > 0) lastUserBarEndTimeApprox = currentBeatInfo.downbeat_times[i]; // End of current or start of next bar
                    else lastUserBarEndTimeApprox = currentBeatInfo.downbeat_times[0] + currentBeatInfo.estimated_bar_duration;
                    break;
                }
                // If user finished exactly on a downbeat, consider that bar 'done'
                if (Math.abs(currentBeatInfo.downbeat_times[i] - userRapEndedInstrumentTime) < 0.1) {
                    lastUserBarEndTimeApprox = currentBeatInfo.downbeat_times[i] + currentBeatInfo.estimated_bar_duration;
                    break;
                }
            }
            if (targetAiStartTime === -1 || targetAiStartTime <= userRapEndedInstrumentTime) { // If loop didn't find good spot
                lastUserBarEndTimeApprox = userRapEndedInstrumentTime + currentBeatInfo.estimated_bar_duration; // Fallback
            }


            // Find the first downbeat strictly after lastUserBarEndTimeApprox minus a small tolerance
            targetAiStartTime = currentBeatInfo.downbeat_times.find(dt => dt > lastUserBarEndTimeApprox - 0.1);


            // If no suitable future downbeat found (e.g. end of track data), fallback
            if (!targetAiStartTime || targetAiStartTime <= userRapEndedInstrumentTime) {
                targetAiStartTime = currentBeatInfo.downbeat_times.find(dt => dt > userRapEndedInstrumentTime + 0.5);
            }
        }

        // Fallback if downbeat calculation failed or no downbeats
        if (!targetAiStartTime || targetAiStartTime <= userRapEndedInstrumentTime) {
            targetAiStartTime = userRapEndedInstrumentTime + (currentBeatInfo.estimated_bar_duration || 2.0); // Default to ~1 bar after current time
            console.warn("Complex downbeat timing failed or insufficient beat info, using fallback AI start time.");
        }

        const delayMilliseconds = Math.max(0, (targetAiStartTime - userRapEndedInstrumentTime) * 1000);

        console.log(`AI Target Instrumental Time: ${targetAiStartTime.toFixed(3)}s. User End Time: ${userRapEndedInstrumentTime.toFixed(3)}s. Calculated Delay: ${delayMilliseconds.toFixed(0)}ms`);
        aiPlaybackStatus.textContent = `AI drops in at ${targetAiStartTime.toFixed(1)}s on the beat!`;

        setTimeout(() => {
            if (beatAudioPlayer.paused) {
                console.warn("Beat was paused when AI tried to play. AI response skipped.");
                aiPlaybackStatus.textContent = "Beat paused. AI response skipped.";
                recordMicBtn.disabled = false;
                return;
            }
            // Ensure instrumental is at the target time (or close enough)
            // This is a safeguard; ideally, the timing of setTimeout is precise enough.
            // beatAudioPlayer.currentTime = targetAiStartTime; // This can be jarring if timing is off. Better to rely on setTimeout.

            aiAudioPlayer.play().catch(e => console.error("Error playing AI audio:", e));
            aiPlaybackStatus.textContent = "AI RESPONDING!";
            console.log("AI Audio Play triggered. Instrumental time:", beatAudioPlayer.currentTime.toFixed(3));
        }, delayMilliseconds);

        aiAudioPlayer.onended = () => {
            aiPlaybackStatus.textContent = "AI finished. Your turn!";
            recordMicBtn.disabled = false;
            userRapProcessed = false;
            aiIdleMessage.style.display = 'block';
            aiResponseContent.style.display = 'none';
        };
        // Fallback re-enable
        const aiAudioDurationEstimate = (aiAudioPlayer.duration && isFinite(aiAudioPlayer.duration) ? aiAudioPlayer.duration : 15); // default 15s
        setTimeout(() => {
            if (recordMicBtn.disabled) {
                recordMicBtn.disabled = false;
                console.log("Re-enabled record button via timeout fallback.");
                if (aiAudioPlayer.paused && !aiAudioPlayer.ended) {
                    aiPlaybackStatus.textContent = "AI timed out or playback issue. Your turn!";
                }
            }
        }, delayMilliseconds + aiAudioDurationEstimate * 1000 + 2000);
    }

    // Initial UI State
    recordMicBtn.disabled = true;
    updatePlayPauseButton(false);
    aiIdleMessage.style.display = 'block';
    aiLoadingDisplay.style.display = 'none';
    aiResponseContent.style.display = 'none';
    liveFeedbackDisplay.style.display = 'none'; // Hidden until beat is loaded
});