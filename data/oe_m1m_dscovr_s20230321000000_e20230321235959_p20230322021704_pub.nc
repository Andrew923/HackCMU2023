CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230321000000_e20230321235959_p20230322021704_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-22T02:17:04.696Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-21T00:00:00.000Z   time_coverage_end         2023-03-21T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxpr�  
�          A33����A�H�C�
����B��H����A(�?�
=A3\)B�#�                                    Bxp�f  S          A
=�h��A\)�C�
���B�\)�h��A�
@�AB{B�{                                    Bxp�  T          A��`��A���p  ����B�3�`��A�?�z�ABݮ                                    Bxp��  
�          A
=�p��A�
�c33��p�B��
�p��A�?�ffA\)B�{                                    Bxp�X  
Z          A\)�N�RA��J�H��Q�B����N�RA
=@�A@��BڸR                                    Bxp��  "          A   ��p�A	p����R�9B�p���p�A�@:�HA��B���                                    Bxpʤ  T          A��w�AQ��p���
B�u��w�A��@W�A��B�q                                    Bxp�J  
�          A��FffAp��$z��o�
Bٮ�FffA��@,(�A{33B���                                    Bxp��  �          A ���_\)AQ��z��@��B�G��_\)A�@E�A��Bߔ{                                    Bxp��  �          A#33�x��A�R��  ��
=B��H�x��Ap�@�=qA��
B�Q�                                    Bxp<  T          A!��g�A�H�\)�Q�B����g�A ��@�p�A��B�k�                                    Bxp�  T          A!��Z=qAff�����ۅB���Z=qA	p�@�Q�A��B�\)                                    Bxp"�  
(          A
=�Az�@b�\A�33B���@��@�=qBO�HB�{                                    Bxp1.  
Z          A   � ��A�H@��A�33B�k�� ��@�(�A33B[�
B�                                    Bxp?�  T          A ���#33A=q@x��A�z�B�\�#33@�{A z�BT��B�G�                                    BxpNz  �          A\)�EAG�@*�HAx��Bٞ��E@�  @�33B6  B螸                                    Bxp]   T          A�
�0��A@�\A=�B��H�0��@ڏ\@�
=B)�B�
=                                    Bxpk�  �          A33�p�Ap�@$z�Ao�BΣ��p�@���@���B833B�k�                                    Bxpzl  "          A�G�@��
@�z�A�(�B�8R�G�@���AG�Bi�HB�#�                                    Bxp�  �          A��C�
A�@�Q�AͅB�ff�C�
@���A ��B[Q�B�ff                                    Bxp��  �          A��[�@ڏ\@�{B
=B�B��[�@\)A{B�
=C�                                    Bxp�^  "          A���@�33@���B  B�#���@@��A(�B�.B�Q�                                    Bxp�  T          A�ÿ�
=@�\@��\BBЏ\��
=@Tz�A�B��B�ff                                    Bxpê  T          A=q���A=q?�z�A7�B�ff���@�(�@�ffB
=C �H                                    Bxp�P  �          A
=���@�\)@8Q�A��HC ����@���@�p�B!  C��                                    Bxp��  
�          A{���@�z�@5A�z�C�����@��\@�=qB��C��                                    Bxp�  "          AG��ʏ\@ڏ\@B�\A���C�{�ʏ\@�p�@ƸRB(�C�H                                    Bxp�B  �          A�����@���@eA��HB�����@��\@�B8�C	�
                                    Bxp�  
Z          Aff��
=@�\)@��HAиRC ���
=@g
=@�(�BA��C��                                    Bxp�  T          A   ���H@�Q�@z=qA�\)B�����H@�  @�BAz�C	�                                    Bxp*4  �          A��|��A\)@P��A�\)B�\)�|��@�\)@�B=ffB�\                                    Bxp8�  �          A\)�;�AG�?�\)A-G�B��)�;�@�p�@���B${B��H                                    BxpG�  T          A   �333A��@ffAB=qB�aH�333@��@׮B)B�R                                    BxpV&  �          A\)�Q�A�@ ��Ak�B���Q�@�  @��B5�B�=q                                    Bxpd�  �          A (��J�HA�H@��HA���B�aH�J�H@�(�A�HBeffB��)                                    Bxpsr  �          A���]p�@�{@�\)B&33B� �]p�@�A�\B�z�C�\                                    Bxp�  T          Ap�����@޸R@�{B
=B�{����@9��A(�Bo  C=q                                    Bxp��  "          AG��)��A(�@�  A�p�Bמ��)��@�(�A
=Bc�B�                                    Bxp�d  	�          A=q��G�@�33@��\A��B�3��G�@�G�@�\)BHz�C�
                                    Bxp�
  �          A\)��p�A�@��
A֣�B����p�@�(�A
=BfB�\                                    Bxp��  �          A����AQ�@��HB��B�#׿��@�ffAffB}33B�W
                                    Bxp�V  �          A33��
=@�{@�(�B
�B��ÿ�
=@qG�A�B��=B��                                    Bxp��  T          A�����R@�  @�G�B@�B�aH���R?��HA�
B���CǮ                                    Bxp�  �          A=q�H��@�ff@S33A�33B�aH�H��@���@��
BHp�B�\)                                    Bxp�H  �          A�R���
A��?�p�A"{B�L����
@љ�@�33B
=B�aH                                    Bxp�  "          A�R���
A��?&ff@qG�B�����
@�p�@��RA��\B�q                                    Bxp�  �          A�H����A	G�?fff@�\)B�(�����@�=q@���B��B��                                    Bxp#:  "          A�H��Q�A��@>{A��B�L���Q�@�ff@��B4��B�B�                                    Bxp1�  T          A�\��=qA�\@"�\An=qB�
=��=q@��@��HB&��C +�                                    Bxp@�  �          A=q����AQ�@UA�B�3����@���@�{B:ffC                                    BxpO,  
�          A�����R@��
@`  A���B������R@��@�B9�C)                                    Bxp]�  "          A(�����@��@��A��B�����@�z�@�\)BM�HC�
                                    Bxplx  
�          A���X��@��@���B��B�G��X��@6ffA��Bzz�C                                      Bxp{  �          A
=���HAp�@l(�A�\)B�k����H@�ff@�z�BA=qC@                                     Bxp��  �          A (����
@�=q?�
=A3�
C �����
@�@��B	�
C
p�                                    Bxp�j  �          A
=�0��@��
@�(�B�B�ff�0��@XQ�A��BvCG�                                    Bxp�  "          A�\�
=@���@�
=B\)B��׿
=@aG�A�RB�� B�
=                                    Bxp��  
�          A�\�p�@�p�@�=qBffB�(��p�@^�RA�B���B�Ǯ                                    Bxp�\  
(          A=q��z�A�
@��
A�p�B����z�@��HA
�RBuG�B�                                     Bxp�  �          A녿޸R@ᙚ@�(�B)�RBϳ3�޸R@"�\Ap�B��\B���                                    Bxp�  
�          A��tz�@�{@�ffA�33B��H�tz�@�\)@��BW�\C!H                                    Bxp�N  T          A  ��G�@�z�@�{B��B����G�@x��A(�B�
=B��                                    Bxp��  
Z          A�ffAQ�@���A�BЏ\�ff@��A{BlQ�B�=                                    Bxp�  �          A����Az�@�z�A���B�.����@�(�A\)Bz\)B�L�                                    Bxp@  "          A���
A��@��A�G�B�{��
@���A��Bg�B�Ǯ                                    Bxp*�  T          Ap��HQ�Aff@�Q�Aڏ\B���HQ�@��A��B]��B�L�                                    Bxp9�  �          A�\�\(�A�@�(�A�33B���\(�@�ff@��BR{B���                                    BxpH2  T          A��EA�R@��Aܣ�B�ff�E@�33AffB_(�B��R                                    BxpV�  �          Aff�2�\Az�@X��A�p�B�L��2�\@��@�RBE�HB��                                    Bxpe~  
�          A"{�7�A�@]p�A�{B�{�7�@�{@�p�BE33B�z�                                    Bxpt$  T          A   ��{A�>L��?���B�
=��{A33@�\)B��B���                                    Bxp��  �          A
=�P��A  ?��@��
B��)�P��@��H@��B=qB���                                    Bxp�p  �          A   �P  A33?�{@�
=B���P  @�G�@�
=Bz�B��                                    Bxp�  �          A#33�EA�@ffA@(�B؀ �E@�p�@��B%�B��                                    Bxp��  �          A"�\�Q�A�@���A�p�Bҏ\�Q�@�33A�\B`�RB�\                                    Bxp�b  T          A!����=qAG�@���A��B�LͿ�=q@�ffA�Be=qB��
                                    Bxp�  T          A"{�$z�A	p�@��Aٙ�B�L��$z�@��RA=qBaQ�B�q                                    Bxpڮ  
�          A ���(�@��R@��\B\)B���(�@z�HA=qBz(�B���                                    Bxp�T  �          A z�aG�Aff@L(�A��B��q�aG�@���@��HBF\)Býq                                    Bxp��  �          A!p��%�A ��@�=qA�
=B�u��%�@�G�AG�Bd�B�.                                    Bxp�  �          A!���|��@�p�@�(�BQ�B� �|��@0��A�BuffC�                                    BxpF  "          A�
���@�@��B	�
B�  ���@N�RA��Bk
=C��                                    Bxp#�  T          A�H�i��@��@ǮB�RB�aH�i��@'�A��B|p�C\)                                    Bxp2�  �          A��qG�@ٙ�@��B�B�
=�qG�@)��Az�By(�C�H                                    BxpA8  T          A�
�l(�@�G�@�
=B
=B���l(�@S33A
�\Bq  C
(�                                    BxpO�  T          A��aG�@�@��Bz�B�Ǯ�aG�@1�A��B|�\C��                                    Bxp^�  "          A�
�AG�@�p�@��A�RBݸR�AG�@�G�A�
Bf{B�8R                                    Bxpm*  �          A   �`  A�@��A���B�aH�`  @�=q@��\BO33B�=q                                    Bxp{�  
Z          A��#�
Az�@J�HA�33BӨ��#�
@��
@�B?��B�aH                                    Bxp�v  �          A33�"�\A
�R@���A�=qBԳ3�"�\@�(�@��BT=qB��                                    Bxp�  "          A!p��W
=@�33@�{B	z�B�R�W
=@g
=Az�Br��C�3                                    Bxp��  "          A!��g
=@�Q�@��
B�HB�Q��g
=@dz�A
�RBn��CJ=                                    Bxp�h  
�          A ���R�\Aff@�Q�AمB���R�\@�{A ��BZQ�B�
=                                    Bxp�  "          A#
=�[�@θR@�  B.��B����[�?�(�A�B�  C+�                                    BxpӴ  
�          A#��[�@�33@�{B��B�B��[�@~{A
=Bkz�C�f                                    Bxp�Z  T          A#
=�l��A\)@mp�A�\)B����l��@��H@�z�BC  B��                                    Bxp�   "          A"�H�a�A\)@   Ac�
B��f�a�@�z�@�33B)ffB��                                    Bxp��  "          A"�H�x��A\)?�
=Az�B��x��@�=q@���B33B���                                    BxpL  T          A$  �mp�A�
?h��@��HB޽q�mp�@�ff@��
B=qB�z�                                    Bxp�  
(          A%G����Aff?�A$z�Bͨ����@�\@�(�B�HB�#�                                    Bxp+�  
�          A&�\�z�A�@z�A8(�B�u��z�@���@ۅB%=qB�                                    Bxp:>  �          A&ff�N�RA\)?�\A��B��
�N�R@�R@θRB�B��)                                    BxpH�  
(          A%G��u�Aff?�Q�A
=B�ff�u�@�\)@�\)B
=B��)                                    BxpW�  
(          A%��~{Ap�?�G�@�ffB�{�~{@���@��
B�B랸                                    Bxpf0  �          A'��vffA�\?   @0��B�ff�vffA�@��HA���B���                                    Bxpt�  "          A(���c33Aff>�=q?�(�Bۀ �c33A=q@�  AB��f                                    Bxp�|  
�          A)G��J=qA!����
��Q�B��H�J=qA
=@��HA�\B�                                    Bxp�"  T          A)G��mp�A\)���{B����mp�A�@^�RA�=qB���                                    Bxp��  "          A*{�\��Ap��Tz�����B܅�\��A�?�{A#\)B�                                      Bxp�n  T          A+
=���A=q��=q���B�=q���A&{?�{@��
Bȳ3                                    Bxp�  �          A,Q���RA!G��@����=qB����RA#�@\)AV{B̔{                                    Bxp̺  
�          A,Q��3�
Ap��QG�����B��H�3�
A"ff@	��A8z�B���                                    Bxp�`  �          A,���H��Ap��qG���{B�B��H��A"�H?�=qA�\B�G�                                    Bxp�  
�          A,���)��A������z�B��f�)��A'\)��G��z�B�p�                                    Bxp��  
�          A,���@  A33��(�����B�=q�@  A$  ?�Q�@��
BԨ�                                    BxpR  T          A,���n�RA���Q��ŅB��n�RA!��?+�@b�\B�z�                                    Bxp�  "          A,Q��R�\A�H�z�H��\)Bڅ�R�\A!�?���@�B�\                                    Bxp$�  T          A,���g�A���|(���Q�B�z��g�A z�?��\@أ�Bۨ�                                    Bxp3D  "          A,���Dz�A���{���B����Dz�A#�
>k�?�  B�k�                                    BxpA�  
�          A-���g
=A   �
=q�8  B۽q�g
=A�@H��A���BܸR                                    BxpP�  
�          A/�
�aG�A%���H�%�Bـ �aG�A\)@�Q�A�(�B�Ǯ                                    Bxp_6  
<          A.�\�3�
@�\)@�Q�B��B��
�3�
@aG�A��B���C �)                                    Bxpm�  T          A/33��
@�33A ��BAp�BՏ\��
?�z�A&=qB��RC	B�                                    Bxp|�  �          A.�\��33@�
=@�{B3��B�z��33@(Q�A&=qB�Q�B��                                    Bxp�(  T          A.�\�޸R@�Q�@�ffB3�B�
=�޸R@*=qA&�RB�{B�=q                                    Bxp��  "          A0z���
@��A�B@��Bԏ\��
@G�A*ffB���C�\                                    Bxp�t  
�          A0�׿�{@��A
=Ba�RB�z��{>�A-B�C&.                                    Bxp�  �          A0�ÿ�
=@ʏ\ABV
=B���
=?�=qA-p�B�{CE                                    Bxp��  
�          A0�׿ٙ�@�{A��BeB�33�ٙ�>��RA.=qB�C)��                                    Bxp�f  T          A0�׿�ff@�p�A�\B`�HB�aH��ff?
=A.=qB���C5�                                    Bxp�  "          A0�ÿ�@��
A33BXp�B��쿕?��A.�HB�ffC	&f                                    Bxp�  
�          A,�ÿ���@�A	��BRz�B˸R����?���A*ffB��C��                                    Bxp X  
Z          A.ff��G�@�
=ABm�B�𤿁G�=�G�A-p�B�G�C-�                                    Bxp�  T          A-���(�@�Q�A�Bj�B��f��(�>aG�A+�
B��)C)��                                    Bxp�  �          A,�Ϳk�@�  A�
Bk�B��H�k�>W
=A,  B�C'�                                    Bxp,J  	�          A.=q���@���A
=Bf�RBϔ{���>�p�A,z�B��
C%0�                                    Bxp:�  "          A.ff����@��RA�Bq�B�����;��A-�B���C:!H                                    BxpI�  �          A333��p�@�  A ��B~�\B֏\��p��Q�A1G�B���CQ{                                    BxpX<  �          A3�
�˅@��\AffBv��B֮�˅��A1B��CD+�                                    Bxpf�  
F          A3\)�޸R@��A�\Bn{B׸R�޸R�L��A1�B�{C5�                                    Bxpu�  T          A2{��G�@�=qABe=qB����G�>�A0(�B��C#(�                                    Bxp�.  �          A1녿�G�@�{Az�Bb=qBЅ��G�?(�A/�
B�(�C��                                    Bxp��  �          A0�ÿ�  @�G�A=qB^�B�녿�  ?G�A.�HB�ffC��                                    Bxp�z  
�          A.�R�(Q�@�
=A��B`B�ff�(Q�>��
A)�B��)C-\                                    Bxp�   	�          A+
=�5@�\)Az�Bfp�B�aH�5��Q�A$��B�8RC5                                    Bxp��  
�          A&ff�Dz�@s33A=qBw�\C �f�Dz῔z�AG�B��CH                                    Bxp�l  �          A(����@�ffA{Be�\B�k���>B�\A$��B��HC/+�                                    Bxp�  T          A*ff���@�33A33Bd�HBߙ����>���A&�RB��{C,�                                    Bxp�  �          A+33�=q@�=qA\)BQG�B�\)�=q?�A%��B��C#�                                    Bxp�^  �          A+��{@�G�A
�HBX�B�8R�{?Q�A&ffB��HC!�=                                    Bxp   T          A-��!�@���AffBJ�RB���!�?�  A'33B���C^�                                    Bxp �  	�          A/\)�.{@ָRA�BB��B�\�.{?�{A'\)B�\C��                                    Bxp %P  
�          A.=q�\)@��
AQ�BE�
B�=q�\)?�\A'33B�k�C��                                    Bxp 3�  �          A.{�=p�@��HAz�BG=qB��=p�?\A%�B�G�C��                                    Bxp B�  
�          A,(��QG�@���Az�BIp�B��f�QG�?�G�A"�\B�  C��                                    Bxp QB  T          A,���Fff@��
AffBD{B��
�Fff?�\)A#�B�=qCc�                                    Bxp _�  
�          A-p��4z�@�p�A  BF�B�k��4z�?�\)A%G�B�{C#�                                    Bxp n�  �          A-��!G�@�\)A�BNG�B�\�!G�?�{A&�HB�L�C��                                    Bxp }4  �          A-��z�@�
=A=qBB�B��
��z�@
�HA(  B�ǮB��                                    Bxp ��  
x          A*=q�G�@љ�A�BF�\B�G��G�?�A#�B�\C	�H                                    Bxp ��  T          A)���@�G�A	�BWQ�B�����?p��A$��B���C�\                                    Bxp �&  
�          A)��@�(�A�RBf(�B晚��>L��A$��B�ffC/O\                                    Bxp ��  T          A,  �  @���A�HB`�
B�W
�  ?��A'�
B�
=C&33                                    Bxp �r  T          A,  ��  @�A�BH
=B�LͿ�  ?�33A&�RB�33C�\                                    Bxp �  
�          A*�\���@��
@�\)B9�\Bѣ׿��@!�A#
=B���B�G�                                    Bxp �  
�          A&�\���H@��
@�p�B'  Bɳ3���H@W
=A�B��3B��                                    Bxp �d  "          A&�H��
=@�=q@���B133B�
=��
=@<(�AffB�u�B�q                                    Bxp!
  T          A'33����@��@�  B7��B��ÿ���@,��A Q�B�.B��
                                    Bxp!�  T          A&�R����@���@ᙚB+(�Bϙ�����@G�A�B�=qB�\)                                    Bxp!V  
�          A$Q�� ��@�
=@׮B$\)B��� ��@UA�
B�W
B�\                                    Bxp!,�  	�          A)���(Q�@��@�=qB \)B��f�(Q�@^�RAffB�W
B�.                                    Bxp!;�  
�          A*�H�
=A ��@��B�\B�k��
=@z=qA
=B��qB���                                    Bxp!JH  
Z          A'���@�
=@��BB��f��@}p�A�HB�G�B���                                    Bxp!X�  
�          A)G��#�
@��H@���B��B�#��#�
@n�RAG�B�p�B��)                                    Bxp!g�  �          A,���%@�ff@�z�B,�\B�L��%@B�\A ��B���CW
                                    Bxp!v:  
(          A,���1�@�(�@��
B%  B�
=�1�@UA{B��\C�=                                    Bxp!��  
�          A-G��0��@��@ᙚB"z�B�33�0��@^�RA{B���C k�                                    Bxp!��  "          A-�?\)@��@�\)B5�B�33�?\)@!G�A"{B�C�f                                    Bxp!�,  "          A,���>�R@�  @��B4�
B�.�>�R@"�\A ��B�k�C�\                                    Bxp!��  �          A/
=�C�
@˅Ap�BG{B�p��C�
?�z�A%��B���Cz�                                    Bxp!�x  �          A0(��E�@�=qA�BI�B�  �E�?�=qA'33B�Q�C�H                                    Bxp!�  �          A0���Vff@�33A�B@33B����Vff?�Q�A%�B�\C�f                                    Bxp!��  �          A1��J�H@�(�Az�BA�B���J�H?���A&�HB��RCc�                                    Bxp!�j  T          A1��4z�@ڏ\A�B@(�B���4z�@
=qA'�B�p�C��                                    Bxp!�  T          A0  ��H@�(�A�BA�B���H@��A(  B�\)C	��                                    Bxp"�  �          A0  ��z�@�
=A��BC�
BҸR��z�@��A)��B�ǮC@                                     Bxp"\  �          A.�H����@���A ��B>{B�  ����@"�\A'�B���B��                                    Bxp"&  "          A0z��#33@�\@�=qB4�
B�W
�#33@4z�A%p�B���C�                                    Bxp"4�  "          A0���4z�@�@��
B.z�B�L��4z�@C�
A#�
B���C�R                                    Bxp"CN  
�          A0Q���\@��@��B9�HB�33��\@.�RA'�B��B�z�                                    Bxp"Q�  �          A0z����@�
=@�\)B2ffB׀ ���@@  A%�B�8RC �=                                    Bxp"`�  
�          A0���   @�R@��B6�RB�{�   @:�HA'�B���B���                                    Bxp"o@  �          A0�׿�Q�@�Q�@��B7(�B�ff��Q�@>{A(  B���B�k�                                    Bxp"}�  �          A1p���Q�@�\)@��RB0G�B����Q�@P��A&�RB�B�ff                                    Bxp"��  "          A1p���R@�ff@�z�B'Q�B�\)��R@g
=A#�
B��B�ff                                    Bxp"�2  
�          A/����A Q�@��
B"Q�BҮ���@s�
A z�B���B�{                                    Bxp"��  �          A.�H��A{@�33B��Bԅ��@���Ap�B�#�B��                                    Bxp"�~  �          A-G���A
=@��HB�\B�����@��Az�Bw
=B��                                    Bxp"�$  "          A.�H��A\)@��Bz�B�  ��@�z�AG�B���B���                                    Bxp"��  �          A/\)�7�A(�@�G�BG�B�G��7�@��A�Bv�
B�=q                                    Bxp"�p  "          A0(���{A ��@��B��B��)��{@��HAG�B^p�C��                                    Bxp"�  �          A/�
��G�A
�H@��
A�=qB��)��G�@��A(�BS��B�8R                                    Bxp#�  "          A/
=�_\)A@��AծB��H�_\)@��HA��BN  B�3                                    Bxp#b  �          A-�W�A��@k�A��
B��)�W�@�G�@���B5
=B�R                                    Bxp#  "          A-p��7
=A�@���A���B�Ǯ�7
=@��A�B_��B�B�                                    Bxp#-�  
�          A-p��L(�AG�@��A�B۸R�L(�@�(�A��B[
=B�aH                                    Bxp#<T  T          A-p��1�A��@���A陚B�.�1�@��
AG�BZ�B虚                                    Bxp#J�  �          A,���!�A\)@��\B  B�aH�!�@��\A�\Bh�HB��H                                    Bxp#Y�  "          A,  �G�A�@�A��B��)�G�@�Q�AG�Bg
=B�R                                    Bxp#hF  T          A+33�$z�A
=q@�\)B�
B�{�$z�@��\A��Bg33B陚                                    Bxp#v�  �          A+33�6ffA��@���A�Q�B׽q�6ff@��A�BZ�B���                                    Bxp#��  T          A+\)�  A�\@�ffB�
B�  �  @�A��Bs��B�u�                                    Bxp#�8  �          A+�
�   AG�@ָRB�\B�Ǯ�   @���AffB��B�aH                                    Bxp#��  T          A,  �P  @�{@�
=B33B���P  @�z�A�Br�C )                                    Bxp#��  �          A-���n�RAff@ÅB	{B�(��n�R@�  A{Bez�C��                                    Bxp#�*  �          A.�\��G�@�
=@ȣ�B\)B�R��G�@���A33Be�RC^�                                    Bxp#��  "          A/33���HA@�{B\)B�=q���H@��A\)B[�C�\                                    Bxp#�v  
Z          A.�R��33A{@���A��HB�z���33@��A�BK��C                                      Bxp#�  T          A.=q��G�A�H@�33A��B�33��G�@�  A��BW\)C �f                                    Bxp#��  �          A/33�a�A
ff@�A�=qB�aH�a�@�A\)B\(�B��{                                    Bxp$	h  T          A+�
���HA(�@���A�Q�B�aH���H@��RAG�BG=qB��                                    Bxp$  �          A+\)����@��@�\)B8�RB�
=����@J=qA#�
B��B���                                    Bxp$&�  
(          A-��Ǯ@�G�@�  B6��B��Ǯ@QG�A$��B��=B���                                    Bxp$5Z  T          A.�R��@�ff@��RB3�HB��q��@\(�A%p�B���B�=q                                    Bxp$D   �          A.�H�0��@�z�@���B.Q�B�  �0��@l(�A$(�B���B�33                                    Bxp$R�  �          A/
=��z�Ap�@�
=B�B��
��z�@�33A\)B�Q�B��
                                    Bxp$aL  �          A/33���A  @θRB��B��쿑�@��RA�HBy{B��)                                    Bxp$o�  �          A-녿�=qA  @�B�B�k���=q@���A(�B�
B�{                                    Bxp$~�  �          A,�ÿ\(�A=q@�Q�B�
B��\(�@�Q�Az�B��Bɮ                                    Bxp$�>  T          A.�H��G�A  @�=qB�RBǙ���G�@�G�A��Bt�B�aH                                    Bxp$��  �          A.�H��
=A
=@��B�RB��)��
=@�ffA��BuB�p�                                    Bxp$��  �          A.�H��p�A��@�\)B{B�8R��p�@�z�A�
Bp��B�33                                    Bxp$�0  T          A/
=��z�A�R@�G�B
=B�#׿�z�@��\ABk=qB�\)                                    Bxp$��  T          A.�H����A��@�=qB �
B�\����@���A\)BfQ�B�                                    Bxp$�|  �          A.�R��{AG�@�ffB33B�  ��{@���Ap�Bk�B���                                    Bxp$�"  �          A.�\��ffA��@�z�B�RB�k���ff@���AQ�Bh��B�=q                                    Bxp$��  �          A.=q��
A{@�\)A�B�ff��
@�G�A
=B\ffB߀                                     Bxp%n  �          A-p��A=q@�=qA�G�BЮ�@�(�A��BX��B�\)                                    Bxp%  T          A-p���RA  @��HA��\B�����R@���A�B_��B�{                                    Bxp%�  �          A,����RAp�@��
A�
=Bϔ{��R@�=qA��BZ�HB��                                    Bxp%.`  �          A,�ÿ�A(�@���A�B����@���A��BY�RB�k�                                    Bxp%=  �          A+�
��A@���A�B����@�z�A�BY{Bܞ�                                    Bxp%K�  �          A,z��9��A�H@���A�RB����9��@��A
=qBUffB鞸                                    Bxp%ZR  �          A,���W
=Ap�@�
=A�33Bݞ��W
=@�{A��BQ(�B�#�                                    Bxp%h�  �          A,  �I��A
�R@���A�\)B�  �I��@��A�
BY�\B��                                    Bxp%w�  T          A,  �L��A@�p�A�{B�Ǯ�L��@��A  BQ(�B�aH                                    Bxp%�D  �          A+��N�RA(�@�=qA��
B�p��N�R@���A�
BI=qB�Q�                                    Bxp%��  �          A+��W�A��@���A��HBܨ��W�@ǮA ��BBQ�B���                                    Bxp%��  �          A+
=�U�Ap�@���Aȣ�B�L��U�@ǮA (�BB33B�=q                                    Bxp%�6  �          A*{���RAz�@��A�=qB�녿��R@�(�A	�BX�
B�p�                                    Bxp%��  �          A)����Ap�@��A؏\B�u����@�z�A  BM��Bޔ{                                    Bxp%ς  �          A(Q���A��@��HA�\)B����@�p�A=qBKffBЮ                                    Bxp%�(  �          A)���z�A��@��
B33B����z�@��RA�Bi(�B�33                                    Bxp%��  �          A'33��z�AQ�@�  A�{B�G���z�@��A�Bb��B�u�                                    Bxp%�t  �          A'33��  A�H@��RA�33B����  @��HAz�BZ�B��)                                    Bxp&
  �          A'33��ffA
=@�33A��B����ff@���A
=BWQ�B�                                    Bxp&�  �          A&�H���HA{@�{A��B��Ϳ��H@�=qA�BYB�Ǯ                                    Bxp&'f  �          A$�׿�=qA�@�\)BffB��=q@�
=A��Bk{B�\)                                    Bxp&6  �          A$(��.{A��@ȣ�B��B����.{@��A�BzBĸR                                    Bxp&D�  �          A$  ��33@��@��
B ��B����33@�Q�A��B��B�aH                                    Bxp&SX  �          A$�ÿ=p�A��@�
=Bz�B�B��=p�@�\)A�HBrG�B�                                    Bxp&a�  S          A)��33A33@���A�G�B�.�33@ə�A�BL�\B��                                    Bxp&p�  �          A)p��33A�H@��AԸRB���33@��A�HBJ\)B�                                    Bxp&J  �          A(���ffA�H@�33A��BШ��ff@�(�A ��BGG�B�aH                                    Bxp&��  T          A)��:=qAQ�@��
A�  B��)�:=q@�p�@���B9��B�.                                    Bxp&��  "          A&�R�(�AQ�@�\)A�{B�Q��(�@�Ap�BKB�                                    Bxp&�<  �          A&�R�{A�R@��A��HB���{@���A33BO��B�aH                                    Bxp&��  T          A'33��A\)@�Q�A�\)Bљ���@�(�AG�BKp�B�33                                    Bxp&Ȉ  
�          A'
=�C33A(�@���A�(�B�W
�C33@�ffA (�BH�B�8R                                    Bxp&�.  �          A'
=�7�A��@�33A�G�B�(��7�@�ffAG�BJ�HB�u�                                    Bxp&��  �          A&�H�3�
Ap�@��A�  B�aH�3�
@Ϯ@���B=  B��)                                    Bxp&�z  �          A&�\�B�\A=q@{�A�p�B�Ǯ�B�\@��@�(�B4�HB�                                      Bxp'   T          A&�\�=p�AQ�@��RA�\)B�L��=p�@�ff@��HB;�RB�G�                                    Bxp'�  �          A&�\�(��A�@���A��Bգ��(��@��HA33BP  B䙚                                    Bxp' l  �          A&ff�/\)A\)@�Q�A�  B���/\)@��HA�RBN�HB�B�                                    Bxp'/  �          A%��6ffA�
@�  Aڏ\B�(��6ff@�\)@�ffBH�B�                                      Bxp'=�  �          A%�@��A��@���AЏ\B�Ǯ�@��@��
@���BC
=B�\)                                    Bxp'L^  �          A%��S33A��@��A�(�B����S33@��H@�{B8�RB�{                                    Bxp'[  T          A#��@  A
{@�Q�A�Q�B�G��@  @�  @�BCp�B�{                                    Bxp'i�  �          A#��N�RA��@�Q�A��B�Q��N�R@��
@���B633B���                                    Bxp'xP  �          A$z��\(�A�@���A�  Bޏ\�\(�@�z�@�G�B4��B�{                                    Bxp'��  �          A%���K�A=q@��RA�Q�B�ff�K�@��
@�B9��B�{                                    Bxp'��  �          A%�� ��A  @���AݮB�� ��@���@�ffBJ�\B�G�                                    Bxp'�B  �          A%�����A�@hQ�A�B�{����@�\)@�z�B&\)B��q                                    Bxp'��  T          A&{���HA  @p��A�  B�#����H@�ff@�Q�B*�B���                                    Bxp'��  �          A&{���\A\)@y��A��\B�33���\@˅@�B-�B�aH                                    Bxp'�4  �          A%���Q�A\)@h��A��\B�#���Q�@�
=@��
B&Q�B��q                                    Bxp'��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxp'�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxp'�&  �          A%���\A33@6ffA�  B�
=���\@ٙ�@�z�Bp�B�Ǯ                                    Bxp(
�  "          A%����A��@*=qAn=qB�{���@�\)@�z�B
��B��
                                    Bxp(r  "          A&ff���HA\)@\)AG33B��)���H@ڏ\@�
=A�Q�CB�                                    Bxp((  T          A&=q��(�A�\@ffAP��B��{��(�@�  @�G�B ��C�{                                    Bxp(6�  �          A&�\��{A(�@(�AB=qB�ff��{@�@��\A���C��                                    Bxp(Ed  �          A&�\���A��@ ��A2�\B������@�  @���A�C ��                                    Bxp(T
  �          A&ff��(�A�?�{A%�B�33��(�@��@�=qA�p�CaH                                    Bxp(b�  �          A&ff��\)A
=@G�A3
=B�W
��\)@�p�@��A�C��                                    Bxp(qV  �          A&�\��=qA
=?�G�A�B�33��=q@���@��A�\)C�H                                    Bxp(�  �          A&�R��=qA\)?�\A��B���=q@ᙚ@�Q�A��
C�=                                    Bxp(��  
�          A'
=����A��@   A_33B�=����@�33@��RB��B�z�                                    Bxp(�H  �          A'�
��G�A�@C�
A�p�B�
=��G�@��
@�=qB{B�Ǯ                                    Bxp(��  "          A(  ��  AG�@Dz�A�{B�{��  @�(�@ʏ\Bp�B�B�                                    Bxp(��  "          A(z�����A�
@�\AIp�B�aH����@�@��\B   B�ff                                    Bxp(�:  �          A(����z�A(�@z�AK\)B�33��z�@��
@��B p�B�8R                                    Bxp(��  �          A)p���  A��@   A.�RB�����  @�G�@��\A��B�\                                    Bxp(�  �          A)��ӅAz�?�@6ffC ���Ӆ@���@p��A�  CǮ                                    Bxp(�,  �          A*�\��\)A�
?�\@.{C@ ��\)@��
@mp�A�\)Ch�                                    Bxp)�  T          A*�\��A�?W
=@��C{��@�\)@���A���C�q                                    Bxp)x  �          A*�\����A�?�@5�C�����@�33@n{A��C��                                    Bxp)!  �          A)����A z�?&ff@`��C�����@�(�@p  A��C�                                    Bxp)/�  �          A)�����HA ��?5@w�CQ����H@�z�@tz�A��HC�=                                    Bxp)>j  
�          A*=q�ۅA?!G�@XQ�C5��ۅ@�
=@p  A�33C��                                    Bxp)M  
Z          A)����33A ��?:�H@|��Ch���33@��
@tz�A��HC�H                                    Bxp)[�  �          A)���33@��ͽ��Ϳ�C����33@��@?\)A��HC+�                                    Bxp)j\  
�          A*ff��z�Ap��\��
Cn��z�@�33@333At(�C33                                    Bxp)y  T          A+���Aff���2�\Cc���@��R@,(�Ahz�C��                                    Bxp)��  "          A/
=��=qA��?���A(�B�u���=q@�@��A���C+�                                    Bxp)�N  
�          A0����=qA�\?��A (�B�����=q@��
@��
Aљ�C�f                                    Bxp)��  �          A.{��p�Az�@%�A\  B����p�@�\@��
B{B���                                    Bxp)��  T          A,z����\A��?�\)@��RB鞸���\A{@�
=A�{B��f                                    Bxp)�@  �          A,����ffA�R?�
=A�B���ff@��@��HA���B��H                                    Bxp)��  �          A.ff���RA{?���@�\)B�{���R@�p�@�\)AθRB���                                    Bxp)ߌ  �          A-���ȣ�A(�?��R@ҏ\B�(��ȣ�@�@�Q�AĸRCxR                                    Bxp)�2  
�          A-���ffAz�>�{?��C #���ff@�  @e�A�Q�C�
                                    Bxp)��  �          A/
=��z�AG�?@  @|(�B��
��z�@��
@�=qA��C�                                    Bxp*~  �          A.�H��=qAp�?��@�\)B�.��=q@�  @�(�A�\)C8R                                    Bxp*$  �          A.�R����A��?У�A	�B�{����@�G�@�
=A��HB�aH                                    Bxp*(�  T          A.{��\)A?�  @��B�L���\)@��@�33A�=qB�33                                    Bxp*7p  �          A-���z�A��?u@��\B�#���z�A (�@��\A�=qB��3                                    Bxp*F  
�          A.{�\A�
?&ff@`  B����\@��H@z�HA�z�B���                                    Bxp*T�  �          A-G�����A	���33���B�p�����A�@�A;
=B��q                                    Bxp*cb  
�          A-�����A���p����HB��
����A
=@C33A�
=B�{                                    Bxp*r  �          A-����
=A���p���(�C ^���
=@��@7
=At��C
=                                    Bxp*��  
�          A-�ʏ\A�?���@��HB��f�ʏ\@�z�@���A�p�C��                                    Bxp*�T  T          A-p���=qA
=?   @*�HB�\)��=qA��@u�A��B�                                    Bxp*��  "          A.ff��Q�A�ͿxQ����
B��)��Q�AQ�@=qAMG�B���                                    Bxp*��  �          A-G���A���\)��p�B��f��A�R@G
=A�Q�B�G�                                    Bxp*�F  �          A-������A  ?�(�@�
=B�R����A�@�33Aȣ�B�z�                                    Bxp*��  �          A-����HA	녿k���B�����HAp�@�AK�B�z�                                    Bxp*ؒ  T          A+
=�޸R@��1G��qG�C���޸RA ��>�  ?��C��                                    Bxp*�8  
�          A)���θRA���p��@z�C ���θRAp�?xQ�@��B��                                    Bxp*��  "          A(�����@�����
�4  C33���@��?z�H@�33Cu�                                    Bxp+�  T          A(����=qA���\)��
=B�B���=qA�?�\)A$z�B��f                                    Bxp+*  �          A(  ��ffA���(���
B���ffAz�?���@�
=B���                                    Bxp+!�  �          A((���ffA�H����A�B�W
��ffA�R?z�H@��HB�Ǯ                                    Bxp+0v  
�          A'�
��Q�A�
>��
?�G�B��
��Q�A   @aG�A�\)B��=                                    Bxp+?  
�          A&�\����A	G��=p����B�#�����A  @�RA\��B�B�                                    Bxp+M�  "          A&�R���
A(�>�\)?\B�\)���
A ��@^�RA�(�B��)                                    Bxp+\h  
�          A'\)���\A
�R>���?�ffB��
���\@�{@_\)A�  B��\                                    Bxp+k  
�          A'33���\A33?n{@��B�=���\@��@���A��HB�p�                                    Bxp+y�  
�          A)���ffA�H?�p�@�
=B�Ǯ��ff@��@��A�=qB�u�                                    Bxp+�Z  
�          A)�����A�@ ��A.ffB�(�����@��R@�A�B�u�                                    Bxp+�   
�          A*ff��(�A
=?�  @�B�aH��(�@��R@�ffA�=qB��                                    Bxp+��  T          A&�R��\)A
{�5�z=qB�33��\)A��@�RA]G�B�L�                                    Bxp+�L  "          A�\��(�A zῴz���HB�
=��(�@��?���A�B�L�                                    Bxp+��  T          A�\���@����(��	�B�u����@�  ?�
=AG�B�ff                                    Bxp+ј  T          A=q���H@�G���Q��4��C �3���H@�\)?k�@��
C B�                                    Bxp+�>  �          A�����@��R�����(�B������@���?��A.ffB�33                                    Bxp+��  T          Ap��˅@�=q���
�33C  �˅@�(�?�(�@�C�                                    Bxp+��  T          A�H�aG�A
=?n{@�  B��aG�A (�@��\A�{B�                                     Bxp,0  �          A=q�XQ�A�
?.{@}p�B�.�XQ�A{@}p�A�33B��                                    Bxp,�  
�          A��   A��?O\)@�Q�B���   A�H@��A�(�B�                                      Bxp,)|  �          A����
A�>L��?���B�33���
A@g�A�G�BǞ�                                    Bxp,8"  �          Aff�$z�Az�<��
=�G�B��$z�A�H@P  A�B��f                                    Bxp,F�  T          A\)�A�A33�#�
�z�HB�k��A�A�R@B�\A�=qBۙ�                                    Bxp,Un  �          AQ��^{A����G��.{B��)�^{A��@C33A���B�\)                                    Bxp,d  �          A{�b�\A�H����R�\B�33�b�\A��@*=qA�ffB���                                    Bxp,r�  	�          A���L��A�H���\���
B�k��L��A\)@{AW33B�W
                                    Bxp,�`  T          A��I��A=q�����{B�  �I��A{?�33A33B�
=                                    Bxp,�  
�          A�R�eAz��G����B���eA  ?�Q�A"{B�                                    Bxp,��  �          A   ��p�A33=L��>��RB�\��p�A@Mp�A�{B�                                    Bxp,�R  
�          Ap���z�A�
�}p���Q�B�ff��z�AQ�@
=qAJ�RB�=                                    Bxp,��  
�          Aff���A�333���B��H���A��@p�Af�HB�\                                    Bxp,ʞ  
�          Ap���ffA(������B�B���ffA
=?��
A'
=B瞸                                    Bxp,�D  T          A����33A�=���?z�B����33@���@C�
A�(�B�(�                                    Bxp,��  
�          A  ��\)AG����
���
B���\)@�?�{A1p�B��q                                    Bxp,��  
�          A�����AG��ff�F�\B�����A��?\(�@���B�3                                    Bxp-6  
�          A
=��A (��G��X��B�33��A��?+�@|��B�ff                                    Bxp-�  
�          A33���\@�Q��	���NffB������\A Q�?.{@���B�33                                    Bxp-"�  
�          Ap�����@��R��p��&�RB�k�����A�?��@ۅB�                                    Bxp-1(  T          A����{@���33�a�B�{��{A ��?�@K�B�                                    Bxp-?�  �          A�����@�Q��ff�J�\CQ����@陚>�@0  C(�                                    Bxp-Nt  �          A�
����@��H��R�r�RC)����@�Q�=�?@  B�                                    Bxp-]  �          A�����@�  ����jffB�L�����@��>���@�B��q                                    Bxp-k�  
�          A���G�@���H�m�B�\��G�@�\)>�{@�B��{                                    Bxp-zf  
�          A�
��=q@���\�_�
B��f��=qA   >�@;�B�                                    Bxp-�  �          AG����
@��H�33�G\)B��=���
@��\?+�@��B��
                                    Bxp-��  
�          A�H�ʏ\@����{�O\)C�ʏ\@�33>��@ffC�\                                    Bxp-�X  �          A����@Ϯ�E���\CǮ���@���0���|��C��                                    Bxp-��  
�          A���Q�@У��/\)��
C	
��Q�@�녾�Q���\C�=                                    Bxp-ä  
�          A���
=@У��w
=��ffC����
=@�{��z��=qC�                                    Bxp-�J  
�          A (��ҏ\@˅������(�C���ҏ\@�33�У���C�
                                    Bxp-��  �          A�H��G�@��H�5���C#���G�@�{����`  C	u�                                    Bxp-�  �          A
=��G�@ƸR�q����RC	���G�@�(������ffC�
                                    Bxp-�<  T          A��߮@�{�A����
C
u��߮@�33�B�\��C��                                    Bxp.�  T          A  ��@���=p�����C����@�׿��H��C}q                                    Bxp.�  �          A�R��
=@����,�����Cff��
=@��Ǯ��
C
                                    Bxp.*.  �          A\)����@�G��%��}�CO\����@�  �u��p�B��                                    Bxp.8�  "          A�H��(�@��R�^{���C���(�@��ÿ��\����C=q                                    Bxp.Gz  "          Ap���
=@���.{��ffC�H��
=@��
���R�\CW
                                    Bxp.V   �          A���ə�@�ff�mp�����C	�)�ə�@Ӆ�˅�=qC�)                                    Bxp.d�  �          A����ff@��R�K���z�C���ff@���G���=qC�{                                    Bxp.sl  T          Aff��z�@��ÿ��33C ���z�@�33?�  @��C }q                                    Bxp.�  T          A���θR@��H�P  ��Q�CǮ�θR@�(���\)��\C	��                                    Bxp.��  "          AG����@Q���G���C"޸���@k������ڸRC�3                                    Bxp.�^  8          A
=����@z�������C$W
����@^{�g����HC��                                    Bxp.�  >          AQ���\)@���R����C"�)��\)@k��[����HCǮ                                    Bxp.��  "          AG����@K������
CL����@�
=�H�����HC#�                                    Bxp.�P  �          A33���
@U��(����CJ=���
@�Q��`  ��=qC0�                                    Bxp.��  T          A����(�@N�R���\��C.��(�@�
=�o\)���C��                                    Bxp.�  T          A����p�@^�R����ffC����p�@��
�X������C��                                    Bxp.�B  �          A  �˅@��R�y����G�C��˅@�  �{�l��C(�                                    Bxp/�  T          A���˅@�(���ff��G�C��˅@�33�6ff����C��                                    Bxp/�  
�          A=q��
=@��H�������HC���
=@�33�<������Ck�                                    Bxp/#4  T          A�H��G�@Vff��
=��\)C���G�@�ff�W����CO\                                    Bxp/1�  
Z          A�R��G�@^�R��33����C�
��G�@�G��N{��\)C�=                                    Bxp/@�  �          A�
��(�@�z��{��ՅC���(�@�{���{�C�3                                    Bxp/O&  �          A	G��ȣ�@G
=��(����C���ȣ�@����g
=��Q�C33                                    Bxp/]�  
�          Az����@{��(��5�HCǮ���@�������33C�                                    Bxp/lr  "          Ap���z�@E��G���RCB���z�@�z���Q���{C�{                                    Bxp/{  
�          A���G�@P  ����
=C�{��G�@�\)�r�\���C�                                    Bxp/��  
�          A����@�Q쾏\)�	��C
���@��?�G�A?
=C(�                                    Bxp/�d  
�          A���p�@B�\����p�C�\��p�@����`����ffC0�                                    Bxp/�
  "          A���{?xQ���33�0ffC*O\��{@.{����\)C�                                     Bxp/��  �          A����
?��H�����{C!�����
@`  ��  ���C.                                    Bxp/�V  "          A ����Q�@�R�_\)����C����Q�@^{�!G����\C��                                    Bxp/��  
�          Ap���  @>{�QG���G�C@ ��  @w
=�	���w�C@                                     Bxp/�  
�          A=q��=q@h���2�\����C����=q@��H��p��*ffC�=                                    Bxp/�H  "          A Q�����@Fff�U���  C�{����@�  �
�H�}p�C�                                     Bxp/��  	�          A ����@=p��R�\��  C���@vff���|��C�                                    Bxp0�  
�          A ����(�?�p��@  ��{C&W
��(�@%�33����C                                       Bxp0:  "          @�\)��
=?޸R�P����G�C%����
=@+��"�\��ffC�                                    Bxp0*�  	�          A z��θR@U�Tz��ď\C���θR@�
=�ff�s�
C�)                                    Bxp09�  
Z          @�
=��G�@�
���\��C�
��G�@c33�XQ��ȸRC�                                    Bxp0H,  
�          A ���\?��H�����
=C(� �\@.�R��(��p�C�
                                    Bxp0V�  
�          A �����H?�(�����4z�C$�{���H@I����=q�z�C��                                    Bxp0ex  T          A�H��{@���������
C�3��{@�p��Fff���B���                                    Bxp0t  "          A33��ff@(���
=�(�C��ff@w��}p�� �HC��                                    Bxp0��  "          A�H��(�@��H������\C����(�@�p��%��\)C	^�                                    Bxp0�j  
�          A���Q�@��H�<(���C�R��Q�@��ÿ�{��RC�
                                    Bxp0�  "          A�����@��\����n�RC�q����@�  ���i��C��                                    Bxp0��  
�          A
=q��  @�Q쿏\)��C���  @�33?�@tz�C�{                                    Bxp0�\  
�          A
�\��(�@�
=��Q��
=C����(�@��\?��
AQ�C�{                                    Bxp0�  T          A=q��z�@�=q�^{���C�3��z�@��Ϳ�p��C�Cc�                                    Bxp0ڨ  �          AG���z�@�����#C	�R��z�@����w
=�޸RCc�                                    Bxp0�N  
�          A����@�{�����{CQ����@�{�G����RC �
                                    Bxp0��  T          A�R����@����  ��33C5�����@�z��#�
��ffCu�                                    Bxp1�  T          A33��33@�G��!G����HC@ ��33@�=q�Y������Cc�                                    Bxp1@  �          A\)��  @��R��\�fffC�f��  @�(�?��
@�\)C)                                    Bxp1#�  T          AG����
@���?E�@���C�f���
@�p�@�
A��C��                                    Bxp12�  
�          A���=q@���?��
@�z�C���=q@�Q�@��A�z�C�                                    Bxp1A2  8          A�
���
@�ff?�\)A
=C�����
@e�@%A��Cu�                                    Bxp1O�  
Z          A{��p�@�
=?#�
@��\C33��p�@��?�(�Ab�HC��                                    Bxp1^~            A����
@��>�Q�@#33C޸���
@�33@
=qAyG�C��                                    Bxp1m$  
�          A�R���@Ϯ�\�,��B�L����@ə�?���A5��B��                                    Bxp1{�  T          A���{@����P  C
=��{@��?��A
=C�3                                    Bxp1�p  �          AG���=q@������mp�Cٚ��=q@�?�G�A�Ck�                                    Bxp1�  �          A=q���@��\���n�RC(����@�  ?��
@陚C�
                                    Bxp1��  "          A����(�@�Q쾳33�\)C����(�@��?�=qA�C	�)                                    Bxp1�b  
�          A{��  @�p�    ���
C	���  @�p�?У�A9CE                                    Bxp1�  
Z          Aff�ʏ\@���>Ǯ@1G�Cu��ʏ\@�{?��AW�Cs3                                    Bxp1Ӯ  
>          A=q����@�
=>�p�@%C.����@�(�?�AQC�                                    Bxp1�T  
�          A
=���H@�33�#�
��\)C�����H@�(�?�Q�A"�\C�f                                    Bxp1��  
(          A�R����@��>.{?�
=C�3����@�p�?���A]p�CJ=                                    Bxp1��  
�          A����@�
=�0����=qC&f���@�z�?���A33C��                                    Bxp2F  T          A���{@��Ϳ��R�d  C\)��{@�
=������C�{                                    Bxp2�  �          AQ��a�@�\)��G��EG�B��a�@�>��@Q�B���                                    Bxp2+�  �          A(����\@ۅ=�G�?@  B�u����\@У�@��Ap��B�.                                    Bxp2:8  T          A����ff@߮>�=q?�B�  ��ff@�33@�A�Q�B�
=                                    Bxp2H�  �          A33��{@��>�p�@(Q�B�=q��{@�z�@�A��B��{                                    Bxp2W�  
�          A(����R@�\)?���A
=B�=q���R@���@Tz�A��B��q                                    Bxp2f*  T          A��g�@�=q?�33A;\)B�(��g�@ȣ�@j=qA�ffB��                                    Bxp2t�  "          @�\)��@8�ÿ޸R�P��C^���@QG��O\)���C��                                    Bxp2�v  �          @�(���\)@7
=�k���C� ��\)@@  �#�
��\)C�                                     Bxp2�  
�          @�  ��?0���?\)���C.}q��?��
�)����ffC'�
                                    Bxp2��  �          @�p����
?&ff�-p���  C.�����
?�z������ffC(��                                    Bxp2�h  
�          @��H���
��Q��%���RCA�{���
�\(��>{���C;#�                                    Bxp2�  T          @���  ��ff�'����C@����  �8Q��<����{C:\                                    Bxp2̴  
�          @�Q���{�
�H�*=q��33CF�H��{��ff�K��Σ�C?k�                                    Bxp2�Z  
v          @�  ���H@����`  ��C=q���H@��
����33C
�                                    Bxp2�   T          @�  ��\)@�33�?\)���C  ��\)@��׿��
�5��C!H                                    Bxp2��  "          @������@G��G��m�C"�R����@0  ���\���CJ=                                    Bxp3L  
�          A Q���Q�@�\��{�8��C#���Q�@*=q�^�R��Q�C }q                                    Bxp3�  T          A ������@G����H��C%.����@�\�z���(�C#Y�                                    Bxp3$�  T          A���@�\��(��\)C#:���@#33�   �fffC!}q                                    Bxp33>  �          @�p����
?�zῧ��z�C'�R���
?�(��G���(�C%��                                    Bxp3A�  T          @�����H@ �׿Tz���=qC%.���H@
=q�aG���{C$
                                    Bxp3P�  �          @������
@ ��<��
>\)C%:����
?�?
=@��\C%޸                                    Bxp3_0  "          @�(����?�=q?�R@��C(Y����?�=q?�ff@�{C*�                                    Bxp3m�  �          @����  ?�p�?�@u�C%=q��  ?�  ?��@��C&�f                                    Bxp3||  
�          @�
=���@Tz�>8Q�?�{CQ����@HQ�?�\)A�HC�)                                    Bxp3�"  
�          @�(�����@R�\>#�
?��RC�3����@Fff?��A\)C�R                                    Bxp3��  
�          @�33���
@A녿z�H��RC8R���
@K����
���C&f                                    Bxp3�n  �          @�p��Ǯ@o\)��33�0��C
�Ǯ@k�?:�H@�\)Cu�                                    Bxp3�  T          @����Q�@H�þ�(��X��CL���Q�@HQ�>�@s33CY�                                    Bxp3ź  
�          @�Q����
@=p���33�3�
C����
@O\)�   ��  C�                                    Bxp3�`  
�          @�=q���\?�z��w����CY����\@;��J�H��  Cff                                    Bxp3�  
(          @�{��(�@h��>�p�@hQ�C����(�@X��?��AY��C��                                    Bxp3�  �          @�=q���@^{�z�����C� ���@`  >�
=@�  C��                                    Bxp4 R  
�          @θR�z=q@�
=����a��C���z=q@��R������C 8R                                    Bxp4�  �          @���Q�@�  �	�����
B�p��Q�@�(���G��j=qB�ff                                    Bxp4�  
Z          @��H�|(�@�G������HB����|(�@�  �\(���Q�B�#�                                    Bxp4,D  
�          @�{�e@4z�<��
>k�C�H�e@,��?O\)A#
=C�                                    Bxp4:�  
�          @�Q��;���z�@���BV�\CI���;���
@q�B5�CZT{                                    Bxp4I�  �          @����\)�@  @��Bu��CF���\)��Q�@~{BU  C\ٚ                                    Bxp4X6  �          @��H��G��Q�@�
=Bn�HCj�Ϳ�G��S33@l��B6Q�Cuu�                                    Bxp4f�  T          @��\��\�   @�B^  C]#���\�J=q@l(�B-Cj                                    Bxp4u�  �          @���L���1G�@S�
B=qC\�H�L���e�@=qA�G�Cd:�                                    Bxp4�(  "          @�����R�E@(��B{Cj#���R�l��?�
=A�(�Cn��                                    Bxp4��  �          @�����\��Q�@G�A�C����\��\)?��AP��C��)                                    Bxp4�t  �          @��H�Ǯ�AG�@�
=B�G�C�H��Ǯ��Q�@��
BF�\C���                                    Bxp4�  
�          @��?�
=�G
=@��
Bw
=C�T{?�
=��  @�Q�B:C�\)                                    Bxp4��  "          @�R?��c�
@�Q�Btz�C�!H?���
=@���B4��C��3                                    Bxp4�f  T          @�33�fff���@��BP��C����fff��(�@���B�RC�>�                                    Bxp4�  
(          @׮��  �~�R@�=qBM\)CyQ��  ��G�@���B�\C~&f                                    Bxp4�  "          @�{�z����\@��B9
=CvǮ�z�����@|��A��\C{ff                                    Bxp4�X  
�          @��\�{��ff@�{BE�\Cp�R�{���@��B  CwQ�                                    Bxp5�  "          @��l(���ff@fffA�  CiL��l(���  @
=A���CmL�                                    Bxp5�  �          @�  ���\��=q@=p�A�ffC�ῢ�\��{?��HAaG�C���                                    Bxp5%J  �          @�p�?�������@!G�A��C�b�?�����33?=p�@�ffC��
                                    Bxp53�  �          @��H?�33���@���B�C�Ǯ?�33�ʏ\@+�A�z�C�^�                                    Bxp5B�  S          @�Q�?B�\���\@�B${C��?B�\����@C33Aʣ�C�c�                                    Bxp5Q<  
�          A��>����@���B6ffC��>����@���A�ffC��                                    Bxp5_�  T          A�
=u��33@���BJC�Y�=u��p�@��\B��C�E                                    Bxp5n�  �          A녾B�\��ff@�  BT(�C��q�B�\��R@�  B(�C�<)                                    Bxp5}.  �          @�zᾣ�
���\@�G�BQ�C�
���
�˅@�  B�
C���                                    Bxp5��  �          @陚�����\)@VffA�{Cv�)�����?�AZ�HCy!H                                    Bxp5�z  T          @��H�7
=��
=?�(�Aa�CuT{�7
=��ff����  Cv�                                    Bxp5�   "          @�G��6ff��33@�A�ffCt��6ff��p�>��R@!�Cv
=                                    Bxp5��  
Z          @����z���ff@�A��HCy��z���33?�@�z�Cz��                                    Bxp5�l  �          @����z����@Q�A�Cxc��z��ƸR>�@�(�Cy��                                    Bxp5�  
�          @ٙ��xQ���ff?�33A�
Cj�\�xQ���녾�Q��A�Ck
                                    Bxp5�  "          @�p��j=q���?�(�AM��Ckk��j=q��Q�#�
���
Clff                                    Bxp5�^  T          @�G���ff����@�{B�\Cz� ��ff����@8��A�z�C}��                                    Bxp6  �          @�{���G�@��BQ�Cu����
=@*=qA�(�Cx�)                                    Bxp6�  
�          @����S�
����@`��A�(�Ck\)�S�
���@ffA��Co33                                    Bxp6P  T          @�33�0����G�@�33BQ�Cn�R�0����G�@>�RA��
Cs��                                    Bxp6,�  
�          @�\�z���33@�z�B.{Cs8R�z���\)@_\)A�33Cw��                                    Bxp6;�  
�          @�׿�
=��=q@��B8Cx(���
=�\@{�A���C|ff                                    Bxp6JB  
�          @�33������@��B@�\Cy^�����\@�{B��C}�H                                    Bxp6X�  �          @�{�{��Q�@�z�BH\)Co�H�{��p�@���B(�Cvs3                                    Bxp6g�  "          @�p��#33�p  @�BW  Ck�=�#33��  @��RB!z�Ct!H                                    Bxp6v4  
�          @�{�#33�Vff@�z�B^Q�ChǮ�#33���@�G�B*(�Cr\)                                    Bxp6��  "          @�(���AG�@�z�BpQ�ChB����p�@�33B<  Csh�                                    Bxp6��  
�          A ���N{�@�=qBv{CT���N{�xQ�@�Q�BL�Cf\)                                    Bxp6�&  "          @��5���{@�B���CM���5��Dz�@\B^  CcQ�                                    Bxp6��  T          AG��!G��#�
@�(�B�W
CB@ �!G��(Q�@�{B|�Cb5�                                    Bxp6�r  "          A	p�� �׾�Q�A33B��C<&f� ��� ��@�=qB�8RCa�                                    Bxp6�  
Z          @����C�
<#�
@�\)B�(�C3�=�C�
���@�\)Bz��CS�                                     Bxp6ܾ  "          A���e�?��@�\B}ffC 5��e��@  @���B��fC?�{                                    Bxp6�d  "          A
=q�]p�@^�R@�33B]\)C�{�]p�?��R@�Q�B�33C�H                                    Bxp6�
  
Z          A��4z�@P  @߮Bip�C�H�4z�?��@��HB�� C{                                    Bxp7�  �          A	G��]p�@��@���Bvz�C@ �]p�=�Q�@�=qB�L�C2p�                                    Bxp7V  T          A���'�@�@���B��C
T{�'�>�\)@�(�B�\C-�                                    Bxp7%�  �          A
{�(Q�@&ff@�Q�B�CL��(Q�>�A�RB�C)��                                    Bxp74�  "          A�R���@Tz�@�33Br  B�ff���?���@�
=B�W
C�                                    Bxp7CH  T          @��H�5@HQ�@��HBeQ�C=q�5?�ff@�B��C^�                                    Bxp7Q�  "          @��s33@K�@���B@��C
�s33?У�@�p�B`�C��                                    Bxp7`�  "          @�\)���@Tz�@��B233Cn���?�@�p�BQ�
C!H                                    Bxp7o:  �          @����@`  @��HB(ffC@ ���@�@��\BJ��Cn                                    Bxp7}�  �          @�\�X��@G
=@�(�BE�C	���X��?���@�Q�Bh  C��                                    Bxp7��  �          @�p��W�@|(�@��
B&��C���W�@'
=@�\)BP�\CG�                                    Bxp7�,  "          @ڏ\�x��@Z�H@��RB"�C
���x��@	��@�{BE�RC#�                                    Bxp7��  T          @���~�R@{�@o\)B(�Ch��~�R@4z�@�(�B-�C��                                    Bxp7�x  T          @�{��=q@��\?Y��@�\)C���=q@n{?��Af{Cz�                                    Bxp7�  "          @�{���
@���@2�\A�Q�C���
@�Q�@}p�B�C	z�                                    Bxp7��  �          @�
=��{@�Q�?��AEp�C����{@�{@1�A��C
\                                    Bxp7�j  
�          @���\)@��H>�33@2�\CY���\)@��\?�33AUC�)                                    Bxp7�  
�          @�G�����@��?c�
@أ�C5�����@�p�@ffA��HC�{                                    Bxp8�  
�          @�Q�����@�G�?��RAG�C�����@���@\)A���C	��                                    Bxp8\  
�          A��ᙚ@\����G���C�3�ᙚ@��QG���G�C�{                                    Bxp8  T          A33��Q�@w
=��=q��\C����Q�@���[����HC�                                    Bxp8-�  �          A����@�����=q��\)Cu����@���?\)���Cc�                                    Bxp8<N  �          A���@��H�>�R���HC�f���@��\�G�C                                      Bxp8J�  T          AQ���ff@�
=�!G����Cp���ff@�ff����C��                                    Bxp8Y�  T          A(���33@��\���tz�C���33@�Q�n{��C�                                    Bxp8h@  �          A����@�(���z����C޸���@θR?�\@G
=C}q                                    Bxp8v�  �          A\)��G�@�����k�
C����G�@�׿�\�N�RC @                                     Bxp8��  T          A=q���H@�Q��p��z�RC�{���H@���(����z�B��                                    Bxp8�2  T          Az�����@�Q��.�R��
=C )����@�\)�k����B��                                     Bxp8��  "          A
=���
@߮�:=q���\B�u����
@�  ����У�B���                                    Bxp8�~  T          A�R��p�@ٙ��e���B��q��p�@��޸R�0  B��{                                    Bxp8�$  �          A�
��33@���|����z�B����33@��H��Q��B�(�                                    Bxp8��  
�          AG���(�@�������(�Ch���(�@�ff�0�����
B��
                                    Bxp8�p  "          A�R���@�ff��xQ�C:����@�33�B�\��\)C\)                                    Bxp8�  �          A��Ǯ@�z�<�>B�\C0��Ǯ@���?�\A/�
C+�                                    Bxp8��  �          A����33@�׿�{�=qB���33@�z�>�@AG�B�#�                                    Bxp9	b  "          A(���\)@�=q�����B�����\)@�p�?�@VffB�.                                    Bxp9  �          A(�����@��5���Q�B�������@�p��}p���G�B�#�                                    Bxp9&�  	.          A�
�љ�@�Q����H��(�C�q�љ�@��R�G����C}q                                    Bxp95T  �          A�����@�����R�jffC	�\���@�z�5��33C�                                     Bxp9C�  "          A���
@������@��C�3���
@�{��\)��G�C�                                    Bxp9R�  
�          A�����
@��\���R���\C�����
@�ff>�=q?�Q�C�                                    Bxp9aF  
�          A=q��{@�G��=p���{C\)��{@���?+�@��CQ�                                    Bxp9o�  �          A�
���R@�  ��\)�"�\C^����R@�  �\�ffC                                    Bxp9~�  T          A���G�@����z��P��C�H��G�@��ͿL������C�                                    Bxp9�8  T          A�H���@�(�?�ff@�(�B������@�z�@3�
A��C�f                                    Bxp9��  �          Az���  @أ�?�@P  C�R��  @�p�@��A]�C:�                                    Bxp9��  	�          A
=��33@�Q��R�r�\C���33@�\)?W
=@�C\                                    Bxp9�*  �          A��=q@>{�6ff��(�C }q�=q@e�G��G�C��                                    Bxp9��  
�          A���?�\�l(���(�C0�\���?��\(����C*��                                    Bxp9�v  T          A��\)��������HC48R�\)?�G���G���  C,�q                                    Bxp9�  �          A���\�ff���\��Q�CC� ��\�u���R��G�C;.                                    Bxp9��  �          Ap���\)@.�R�Fff��(�C!H��\)@Z�H�z���ffC+�                                    Bxp:h  "          A�\��
=@�녿�(��G33B�  ��
=@�논��
��B�Q�                                    Bxp:  
�          A�����
@����3�
���\B��)���
@�  �h����Q�B��                                    Bxp:�  "          A{��@�=q���H��G�C�=��@�������y�C 0�                                    Bxp:.Z  T          A����@��
�\)��z�C���@��=q�u��CaH                                    Bxp:=   
�          AQ���G�@����Mp���C���G�@�z��
=�0  C(�                                    Bxp:K�            A�\����@�G�>W
=?�\)B�aH����@�Q�@ ��AU�B�u�                                    Bxp:ZL  
�          A�\��Q�@�\)�\��HC L���Q�@�33?�\)A��C �\                                    Bxp:h�  
�          A�����@�=q�s33���C8R����@��H?G�@�ffC!H                                    Bxp:w�  	�          A=q���
@�ff�Q���p�C8R���
@�?n{@�=qCE                                    Bxp:�>  �          A�H��\)@أ׿���� Q�Cz���\)@�ff>�?Tz�C��                                    Bxp:��  �          A
=���@��
>L��?�p�B������@��H@G�AM�B��R                                    Bxp:��  �          A�
����@�=q�:�H���\B������@���?���@�(�B�p�                                    Bxp:�0  �          AQ����R@߮���
��(�CxR���R@��?:�H@��CQ�                                    Bxp:��  "          A\)��G�@�(�>W
=?��C���G�@�33?�(�AFffC5�                                    Bxp:�|  	�          A\)���@�33���8Q�C�=���@���?��A$��CQ�                                    Bxp:�"  �          A33��Q�@����Q��  C ����Q�@���?�A=qCQ�                                    Bxp:��  
�          A���  @�Q쾞�R��C�{��  @��
?�
=A�HC&f                                    Bxp:�n  �          A  ��@ڏ\��R�vffC!H��@أ�?��@�Q�CaH                                    Bxp;
  �          A\)��z�@��?��@p��C����z�@��@z�Ai�Cz�                                    Bxp;�  �          A�R�ƸR@�\)>�  ?���C���ƸR@θR?�33A@  C޸                                    Bxp;'`  "          A\)����@�ff?0��@�G�C)����@ʏ\@z�Ai��C�                                    Bxp;6  "          A����
@�G�?�
=A��C�{���
@�\)@C�
A��CL�                                    Bxp;D�  
�          A�H��Q�@�=q?��H@�33Cff��Q�@��@7
=A�33C�)                                    Bxp;SR  T          Az���\)@���>�p�@�C@ ��\)@�33@   AF�RC�\                                    Bxp;a�  
q          A  ��ff@�33>��@!�CY���ff@ə�@G�AJ{C�3                                    Bxp;p�            A=q���
@�(�@
=qA\z�B�=���
@��
@~{A�{B��H                                    Bxp;D  
�          A�\����@�  @\)A~�\B�������@���@�=qA�G�B�p�                                    Bxp;��  �          A�R��p�@�ff?���A:ffB�L���p�@�  @p  A�G�B�                                    Bxp;��  "          A33���R@���?�z�AG�B�����R@��@S�
A�
=B���                                    Bxp;�6  "          Aff���\@�(�?�p�A��B�\���\@��@W�A��
B�                                     Bxp;��  "          A���h��@�Q�@p  A�ffB�Q��h��@Å@�p�B33B�                                    Bxp;Ȃ  "          A���vff@��@7
=A���B�ff�vff@�
=@���A�  B                                    Bxp;�(  �          A�����H@�p�@8��A���B�{���H@�\)@��
A�G�B��                                    Bxp;��  T          A�H����@�=q��G��AG�B�B�����@޸R?��A33B�\                                    Bxp;�t  
�          Ap���G�@�{����� ��C�)��G�@����\��RC ��                                    Bxp<  
(          A���p  @������0��Cff�p  @��\��{��RB�p�                                    Bxp<�  
Z          A����\)@�Q���\)�z�B�����\)@���Z�H����B�q                                    Bxp< f  
�          A
{�U@�  ��
=�LC 33�U@�Q���
=��\B�33                                    Bxp</  
�          Ap���G�@�������z�B�p���G�@�  �s33��ffB�                                      Bxp<=�  �          A33��
=@���E��33B�=q��
=@�{��
=��(�B�Ǯ                                    Bxp<LX  
�          A{�~�R@�{��(��H��B�Q��~�RA�H<�>B�\B���                                    Bxp<Z�  T          A  �j=qA �׿����$Q�B���j=qA�H>�
=@*�HB�#�                                    Bxp<i�  �          A������Aff>���@   B�z�����@���@Q�Ar�HB�z�                                    Bxp<xJ  	�          A����p�A
=?�A$��B���p�@���@hQ�A���B�                                      Bxp<��  T          A(��j�HA�@ ��A|��B��)�j�H@�R@��A���B��                                    Bxp<��  �          A33�HQ�A(�?�G�A\)B�ff�HQ�@�(�@c33A��B�W
                                    Bxp<�<  
Z          AQ��]p�A33?s33@��B�3�]p�@�
=@:�HA��\B�8R                                    Bxp<��  
�          A(���z�@�z�?��Az�C�{��z�@��
@8Q�A��C�f                                    Bxp<��  
�          A���ff@�z�?k�@��C(���ff@�
=@p�A
=C)                                    Bxp<�.  
�          A\)�(Q�A(�?\(�@��
B�aH�(Q�Az�@>�RA�\)B�.                                    Bxp<��  
�          Aff���AQ�?�{A#33B�  ���A@mp�A��\B���                                    Bxp<�z  T          A��A�H?�\@L(�B�33��Az�@+�A��HB�(�                                    Bxp<�   
�          A�R�0��A��?z�@g�B����0��A
=@2�\A�  B�                                    Bxp=
�  T          AQ���A\)?@  @�Q�B��H��A(�@7
=A�=qB�G�                                    Bxp=l  T          A\)�HQ�A��?Y��@��B�=q�HQ�@�33@7
=A��B�p�                                    Bxp=(  �          Ap��{A��>aG�?�\)B�G��{A�@
=Aqp�B�Q�                                    Bxp=6�  �          A����RA�?(��@�
=Bнq��RA��@1�A�\)B�(�                                    Bxp=E^  
�          A�
���A	��?(��@�  B�Ǯ���A�R@0  A���B�W
                                    Bxp=T  �          A33��\)@�=q�#�
���
B�����\)@��H?�A>ffB��=                                    Bxp=b�  
�          A
=���
@�
==�Q�?��B�.���
@�\)?�z�AF{B���                                    Bxp=qP  "          A�����
@�  ��33�33B�Q����
@�?��RA�
B�L�                                    Bxp=�  T          A������@�
=�k����B�z�����@�
=?p��@�z�B�                                     Bxp=��  
(          A=q����@��R�p�����
B��f����@��R?xQ�@�G�B��                                    Bxp=�B  
�          A�����@��H�&ff��{B������@��?�p�@�\)B��                                    Bxp=��  �          A�����\@�zᾏ\)����B��H���\@߮?�p�A33B�
=                                    Bxp=��  
�          A
�R���@�z�@�p�B
�B������@�Q�@���B5��CT{                                    Bxp=�4  T          A����=q@�Q�@�z�BC \��=q@���@��HB6
=C&f                                    Bxp=��  
(          A�H�\)@�p�@���B=qB�W
�\)@n{@��BFffC	\                                    Bxp=�  �          A�H�hQ�@�=q@��B'��B�.�hQ�@b�\@�BS{C�                                    Bxp=�&  
�          A=q�N�R@��@�p�B�B�W
�N�R@��\@�G�B8ffB��\                                    Bxp>�  T          A=q�,��@�=q@c33A��HB�Ǯ�,��@��@�p�B�HB�                                    Bxp>r  "          A��J�H@��H@�{B�\B�\�J�H@�ff@�
=BB��B�                                      Bxp>!  �          A���?\)@�=q@�B!33B�\�?\)@x��@�=qBP��B���                                    Bxp>/�  
�          Ap�����@θR��=q�{B�33����@��H>���@�B�.                                    Bxp>>d  �          Aff�e�@�\)��{�9�B�W
�e�@��>��?�ffB�#�                                    Bxp>M
  <          A����@׮?@  @�  B�����@˅@Q�A�\)B�Q�                                    Bxp>[�  �          A
=���
@���?�ffA
=B��=���
@�(�@5�A���C�)                                    Bxp>jV  T          A33�i��@�{@(�A��HB��)�i��@��
@�(�A���B��                                    Bxp>x�  �          A�H�j�H@�  @9��A�33B�\)�j�H@�=q@���B(�B�L�                                    Bxp>��  T          A��q�@��H@L��A�  B�Ǯ�q�@�33@���B	�HB��3                                    Bxp>�H  
�          A(��fff@���@Z�HA���B���fff@�33@�Q�BQ�B�=q                                    Bxp>��  T          A	G�����@���@QG�A�\)B��q����@���@���BQ�B��{                                    Bxp>��  T          A	���p�@�=q@a�A��B��H��p�@�Q�@�\)B�C�{                                    Bxp>�:  
�          A
{��  @�(�@`��A�  B�W
��  @�=q@���B�RB�                                    Bxp>��  T          Aff�qG�@��H@uA�ffB�\�qG�@�p�@��RBz�B���                                    Bxp>߆  
�          AQ��g
=@��@\)A���B���g
=@�
=@�=qB��B�z�                                    Bxp>�,  
�          A��hQ�@�p�@��\B=qB���hQ�@���@�(�B:  C �H                                    Bxp>��  �          A���`  @���@�{B"�B�#��`  @q�@�=qBO�C�{                                    Bxp?x  
(          A ���,(�@�z�@���BHG�B���,(�@/\)@�G�Bu�\C��                                    Bxp?  �          A�H�Tz�@�G�@��B��B���Tz�@�
=@�{B;z�B�=q                                    Bxp?(�  �          @�33<��
?�@�p�B��B��<��
=#�
@�p�B�\)BmG�                                    Bxp?7j  �          @��>��@E�@�G�B�p�B��>��?�  @�\B�aHB��H                                    Bxp?F  �          @�zῸQ�@�ff@���BU=qB��ÿ�Q�@1G�@�B�
=B�(�                                    Bxp?T�  T          A �ÿ�=q@���@�\)B5��B�33��=q@|(�@��Bm��Bҽq                                    Bxp?c\  T          @��
�
=@�(�@ƸRBTp�B�p��
=@<��@��B�p�B�u�                                    Bxp?r  "          A ���|��@��H@q�A�B�33�|��@�\)@�33B�\C�                                    Bxp?��  
�          Aff���@Ǯ@  A��B�W
���@��@n{A�{C��                                    Bxp?�N  T          A(���(�@���?}p�@���B�����(�@�=q@,(�A�33B�G�                                    Bxp?��  T          A�R��ff@ۅ?^�R@�p�B��f��ff@�{@"�\A�=qB�8R                                    Bxp?��  T          A�\���R@�  ?�33A;33B�����R@�z�@QG�A���B���                                    Bxp?�@  T          A z��HQ�@�\?�@p��B�L��HQ�@�
=@z�A��B�ff                                    Bxp?��  �          A   �z�H@ٙ�?�ffA�\B���z�H@�Q�@<(�A��B�\                                    Bxp?،  T          @��H�Z=q@��H@�A�B��Z=q@���@{�A�ffB��H                                    Bxp?�2  T          @����u�@�ff@J=qA�ffB���u�@��@�G�B=qB�{                                    Bxp?��  T          @���{�@�ff�?\)���
B��f�{�@�  ���)�B�G�                                    Bxp@~  
�          AG�����@���fff��=qCc�����@����
��{C=q                                    Bxp@$  "          A=q��{@s33��
=��RC�\��{@���w���C�{                                    Bxp@!�  �          @�����
=@����  ��C�{��
=@�Q��:=q��{B��=                                    Bxp@0p  
          @�����@*=q��
=�  CaH����@g��[���
=C                                      Bxp@?  �          Az���G�@���
=q�tQ�CY���G�@��׿0������C�
                                    Bxp@M�  �          A�\��G�@ə��!G����CJ=��G�@�Q�?u@У�Cz�                                    Bxp@\b  �          A���z�@�\)?!G�@��
C n��z�@��
@{Ax��C
=                                    Bxp@k  �          A�H����@�p�?�z�A33B��\����@�p�@2�\A��B�                                    Bxp@y�  �          A�z�H@���?u@�G�B�8R�z�H@�=q@+�A�z�B�{                                    Bxp@�T  �          A (���@��@7�A�33B׳3��@��
@�G�B=qB�                                    Bxp@��  �          AQ쿗
=@��
@z=qA�Bƽq��
=@�p�@�=qB*B�u�                                    Bxp@��  �          A(��1�@���@.{A��B�z��1�@Ӆ@�Q�B �
B�\                                    Bxp@�F  �          A	�%@�(�@/\)A�  B�z��%@ָR@��Bz�B�8R                                    Bxp@��  �          A
=q��Q�@�{@��
A�p�B�#׿�Q�@�@�33B+��B�Ǯ                                    Bxp@ђ  �          A���
=@�{@|(�A֏\B�p���
=@θR@��B$z�BȮ                                    Bxp@�8  �          A
�\��33@�z�@p  A�G�B����33@θR@���B �HB�33                                    Bxp@��  �          AQ��@�Q�@{�A֏\B��)��@У�@�  B%(�B�W
                                    Bxp@��  �          A녿�R@�
=@u�A̸RB��
��R@�Q�@��RB =qB�k�                                    BxpA*  �          A��#�
A ��@x��A�B���#�
@�=q@�G�B �
B�                                    BxpA�  �          A(���ffA33@l(�A��B�W
��ff@�Q�@�(�B�B�ff                                    BxpA)v  �          A(��ǮA=q@uA��
B�� �Ǯ@��@�Q�B�B�u�                                    BxpA8  �          A�?uA{@
=qAeB��H?u@��@�ffA��B�z�                                    BxpAF�  �          A?У�A ��@I��A��B��?У�@�Q�@��\B�
B�Ǯ                                    BxpAUh  �          A�?���@�ff@�p�A�z�B�ff?���@���@�\)B'B�=q                                    BxpAd  T          A�H?�33@޸R@�33B�B�8R?�33@�Q�@��BW��B�W
                                    BxpAr�  �          A�H?�ff@�33@��HB�B�z�?�ff@��\@ָRBC\)B�33                                    BxpA�Z  T          A
=?��H@ۅ@���BQ�B�33?��H@���@ۅBJ
=B��                                    BxpA�   �          A�\@�\@�@��B"��B���@�\@�Q�@�BY��B��{                                    BxpA��  �          Aff?�@�(�@�p�B'�B�\)?�@�p�@�33B^��B��3                                    BxpA�L  �          A�?��
@���@��HB6
=B���?��
@�\)@��Bn=qB�\)                                    BxpA��  �          A\)?�z�@�Q�@��HB=Q�B�B�?�z�@y��@��\Bs�HB�                                    BxpAʘ  �          Az�@7
=@��
@�=qB<Bw�
@7
=@a�@�
=Bn{BK�H                                    BxpA�>  �          A��?���@���@xQ�AΏ\B�(�?���@��@�Q�B!{B��\                                    BxpA��  �          Az�@%@��R@ӅB>�B�B�@%@g
=@���Bq(�BY33                                    BxpA��  �          A��@<��@e�@�(�Bgz�BJ�@<��?�=qA ��B��)A��                                    BxpB0  �          AG�@z�@`  @��
Bt�\Bb{@z�?���A(�B�ffA��                                    BxpB�  �          A?8Q�@8Q�A	��B���B�?8Q�?
=A��B�L�Bff                                    BxpB"|  �          A�H?��@{�@�G�Bu\)B���?��?�Az�B�
=B?�                                    BxpB1"  �          AQ�@
=@a�@�z�Bs�RB`��@
=?��HA��B���A�Q�                                    BxpB?�  
�          A	G�?�ff@S33@�=qB~=qBuQ�?�ff?�G�A�\B���Bff                                    BxpBNn  �          Aff?�z�@�33@��
BN(�B��\?�z�@C�
@���B���B��                                     BxpB]  �          A�R?�
=@�33@�Q�BY��B��=?�
=@.{@��RB��\By=q                                    BxpBk�  �          A�?��
@���@�
=Bi�HB�z�?��
@A��B��{Bt��                                    BxpBz`  �          A�?�@��@��BI�B�aH?�@\��A�B�=qBw�                                    BxpB�  �          A�
@33@���@�z�BH��B��q@33@Tz�A (�B|�B\��                                    BxpB��  �          A\)?�p�@���@�\)BC�RB�p�?�p�@fff@�p�Bz=qBt��                                    BxpB�R  �          A�
?�(�@���@ָRBJffB��?�(�@P  @��B�#�Bj�                                    BxpB��  �          A��@9��@u�@�33BcG�BS\)@9��?�Ap�B�.A��                                    BxpBÞ  �          A�\@y��@-p�@��B^  B33@y��?Tz�@��
BvffAAp�                                    BxpB�D  �          A�@�z�?�Q�@��BJ��A�{@�z�    @���BT�C��q                                    BxpB��  �          @�p�@���@33@��B+Q�A��@���?W
=@���B<G�A�                                    BxpB�  �          @�33@>�R@�\)@�G�B;��B[=q@>�R@)��@�ffBh��B&�\                                    BxpB�6  
�          @��H@6ff@�@�{B7��Be
=@6ff@7�@��Bg  B4��                                    BxpC�  �          @��@)��@��R@�  B0��Br�@)��@L(�@�G�Bbp�BI=q                                    BxpC�  �          @��?h��@���@�ffB(�B�  ?h��@��@�G�BPp�B���                                    BxpC*(  �          @�Q���@�p�@0  A�ffB�����@�\)@�z�B��B�z�                                    BxpC8�  �          @�
=�#�
@�ff@z�HA�ffB�8R�#�
@��R@��RB9�\B�B�                                    BxpCGt  �          @�p�?Ǯ@��@���B
�RB���?Ǯ@��@��BD�
B�                                    BxpCV  �          @��?�p�@��@e�A���B��{?�p�@�z�@��B(��B�u�                                    BxpCd�  �          @�{?���@�(�@�z�B�HB�k�?���@�ff@\BI(�B�\                                    BxpCsf  �          @��
����@�
=@P  A�z�B�p�����@�(�@��RB�B�z�                                    BxpC�  �          @�
=���@�  @��A�B��׾��@��R@��
B<�B�Ǯ                                    BxpC��  �          @���R@c�
@�B:{B�.��R@G�@��RBj(�CxR                                    BxpC�X  �          A��0��@�@��HB7�HB�u��0��@N{@�Bj�C��                                    BxpC��  �          A���j=q@�\)@ffA���B�q�j=q@˅@�p�A���B��                                    BxpC��  �          A{�x��@�  ?�{A�\B�.�x��@�z�@J=qA��HB�                                    BxpC�J  �          @�p��aG�@ۅ?�ffA6ffB�B��aG�@ƸR@S�
A���B�
=                                    BxpC��  T          @��n{@ƸR>k�?�=qB��
�n{@�?�{An{B�33                                    BxpC�  �          @��g�@���@�A�ffB����g�@�33@e�B��CxR                                    BxpC�<  �          @��R����@��@�p�B�B�\����@���@ǮBZG�B�p�                                    BxpD�  �          @����(�@�(�@�=qB,B�(���(�@vff@��Bi�HB�Ǯ                                    BxpD�  �          @��Ϳ&ff@��\@��RB��B��q�&ff@�{@�33BQ�BĮ                                    BxpD#.  �          @�?s33@�\)@��B'B�B�?s33@|��@ȣ�Bd{B��f                                    BxpD1�  �          @�\)�\)@�{@���B6\)B��\)@e@ϮBs��B�\)                                    BxpD@z  �          @�ff����@��@���Bz�B��þ���@�=q@�p�BW{B��f                                    BxpDO   �          @�Q�.{@�  @�  Bp�B�\�.{@�33@��BY��B�Ǯ                                    BxpD]�  �          @��\��
@�\)@��HBB{B�����
@@��@�z�Bx�B�Ǯ                                    BxpDll  T          @��;�(�@���@�p�B%�B�G���(�@u�@��Bc��B�                                    BxpD{  T          @�(��0  @�Q�@y��A홚B�aH�0  @��@�B.�\B���                                    BxpD��  T          @��R��@�  @n{A���Bڣ���@���@�=qB+Q�B���                                    BxpD�^  �          A
=����@޸R@.{A�G�B�z�����@�
=@��A�
=B��=                                    BxpD�  �          A����H@�@p�A�  B��H���H@�Q�@�\)A�  C �{                                    BxpD��  �          Aff��33@�Q�?�33AG\)C ^���33@�  @i��A�p�C��                                    BxpD�P  �          A�R��  @��@'�A�p�B�(���  @���@��A��C )                                    BxpD��  �          A�
��\)@ٙ�?�{@��
C{��\)@�  @9��A�G�Cz�                                    BxpD�  �          A
=���R@���?�Q�A.ffC�f���R@�ff@Z�HA�=qC�{                                    BxpD�B  �          A
=��p�@�G�?��H@���C�)��p�@ƸR@@  A�
=Cc�                                    BxpD��  �          Aff����@陚>��H@J=qB�W
����@�z�@��A�B�k�                                    BxpE�  �          A=q��(�@�\)>Ǯ@#33B��q��(�@ۅ@Atz�B���                                    BxpE4  
�          A{��ff@��>�@<��C33��ff@�{@{Ag�
C�)                                    BxpE*�  �          A{��ff@�  >L��?��
B�Ǯ��ff@�@AZffC0�                                    BxpE9�  �          Aff���
@�z�>u?�ffC}q���
@ʏ\@�\AT��C�H                                    BxpEH&  �          A�H��p�@�(����\��C
=��p�@���?�(�@��CxR                                    BxpEV�  �          A
=���H@���;���G�C	
���H@Ǯ���\���CG�                                    BxpEer  �          AQ���\)@�ff?��@��C\��\)@�p�@1�A�33C�                                     BxpEt  �          A	���G�@�?�=q@�
=C���G�@�z�@2�\A�z�CE                                    BxpE��  �          A  ��Q�@�Q�?Tz�@�
=C���Q�@���@%A��C��                                    BxpE�d  �          A���z�@�\)?���A\)C �3��z�@Å@G�A���Cn                                    BxpE�
  �          A�����@�(�?n{@�  C ������@��
@.{A�ffC�f                                    BxpE��  �          A�\��p�@�Q��=p�����C����p�@�33��G��C&f                                    BxpE�V  �          A(��ʏ\@�{��=q�B{C
��ʏ\@�
=�B�\��G�C�                                    BxpE��  �          A  �ƸR@�z��3�
��p�C� �ƸR@�{������33C��                                    BxpEڢ  �          A�R�˅@�ff���H�'\)Cz��˅@��>W
=?��C��                                    BxpE�H  �          A�
����@�Q�.{���C�����@�?�(�@��Ck�                                    BxpE��  �          A�\��Q�@߮�u��Q�C 8R��Q�@�\)?��AC�CE                                    BxpF�  �          A�\��G�@���>�?J=qB�����G�@�ff@��AX��C �\                                    BxpF:  �          A(���(�@�
=?��H@�Q�B��)��(�@�33@J�HA��\Cu�                                    BxpF#�  �          A��G�@��@33AJ=qB�����G�@��@\)A��C�{                                    BxpF2�  �          AQ���@�{=���?�RB�.��@��
@ffAW�C Q�                                    BxpFA,  �          A(���p�@��?#�
@���B�\)��p�@�{@,(�A�  C �                                    BxpFO�  �          Ap���\)@�G�@q�A�
=B�B���\)@�{@�p�BQ�B��                                    BxpF^x  �          A  ��@�Q�@S33A�=qB�33��@���@���B�HB�B�                                    BxpFm  �          Ap����@�\@3�
A�{B�.���@�\)@�=qA���B�ff                                    BxpF{�  �          A���G�@���@�AMB�(���G�@У�@�=qA�
=C
                                    BxpF�j  �          A=q����@����=q�\)C�)����@ڏ\>�Q�@��C)                                    BxpF�  �          A\)�ٙ�@Å��H�m�C
\�ٙ�@�G�����g�C
                                    BxpF��  �          A\)��(�@�p��8Q���=qCG���(�@Ϯ����θRC	��                                    BxpF�\  �          A
=�陚@�\)�Mp�����C
�陚@�p����
�G�C�\                                    BxpF�  �          A(���\)@�33�^�R���\CE��\)@�33��  �$��C
�3                                    BxpFӨ  �          AG���
=@����N{���C:���
=@�
=��Q���C
)                                    BxpF�N  �          A����R@��������
=CǮ��R@��
�y����C\)                                    BxpF��  �          Az����@�����z��  C�����@��H��
=���HC�f                                    BxpF��  �          AQ���Q�@xQ���=q�\)C  ��Q�@�������C�{                                    BxpG@  �          A�
��=q@l�����
��\C����=q@�
=��=q��Q�C�=                                    BxpG�  �          AQ��Ǯ@%��Q��4
=C� �Ǯ@�������\C�                                     BxpG+�  T          A���Q�?�\)���U�HC)���Q�@U����>
=C�f                                    BxpG:2  �          A{��(�?�����\�Hz�C%����(�@k�����.Q�C�                                    BxpGH�  T          A���Q�?��H����:  C#=q��Q�@|(��ə��z�C�{                                    BxpGW~  �          A����@{��p��CffC�{���@�  ��z��!�
C�                                    BxpGf$  �          A���(�@=p�����G�CQ���(�@�  ��33� �\C
h�                                    BxpGt�  �          A����H@&ff��R�=�
C�H���H@�=q������C
                                    BxpG�p  �          Ap��Å@>�R��33�8�HC  �Å@�����p�C=q                                    BxpG�  �          A����p�@(���\)�5�
C8R��p�@�33��
=�ffCٚ                                    BxpG��  �          A���˅@5���3p�C��˅@�\)�����CY�                                    BxpG�b  �          AG���(�@fff����C���(�@��������=qCxR                                    BxpG�  �          A�\��z�@p  ����\)C�=��z�@���{�C�                                     BxpG̮  �          A{�ə�@��H�����!��C  �ə�@�����p���C	xR                                    BxpG�T  �          A���ʏ\@w
=��(��"�HC�H�ʏ\@�����
=��\C
��                                    BxpG��  �          Az���p�@[������'�C�f��p�@�{�����\C�                                    BxpG��  �          A(��Ϯ@0  �׮�.�C\�Ϯ@�33��z��Q�C�f                                    BxpHF  �          A����
=@U���G���C�H��
=@����������C.                                    BxpH�  �          A(����@o\)���
�Q�C33���@����Q��ܸRC�                                    BxpH$�  �          A(���@mp���Q��

=Cs3��@�p���p���33C��                                    BxpH38  �          A  ��
=@��������
C�{��
=@ȣ��k����C��                                    BxpHA�  �          A���z�@��
��{��C����z�@�Q��p����ffC�{                                    BxpHP�  �          A�
����@�{��p��ڣ�C�H����@�
=����eCB�                                    BxpH_*  �          A�
����@�������=qC\)����@����U����\C�3                                    BxpHm�  �          A{��
=@�
=���
� ��C�\��
=@����hQ�����Cu�                                    BxpH|v  �          A�
��(�@w
=���H�Q�C����(�@�33��{��p�C!H                                    BxpH�  �          AQ���33@=p���{�"Q�C�\��33@���������C��                                    BxpH��  �          A��ָR@l��������C#��ָR@�{��{��ffCJ=                                    BxpH�h  T          A�
����@�
=���R��C&f����@���E��p�C
�f                                    BxpH�  �          A�R��G�@�G���ff��ffCc���G�@�z��.�R��p�C�\                                    BxpHŴ  �          A�У�@�Q���33���CT{�У�@�Q��QG�����C
��                                    BxpH�Z  �          A�
���
@�=q���H�癚C&f���
@θR�1���=qC�H                                    BxpH�   �          A=q��  @p  ���H��C��  @�Q���ff��G�C                                    BxpH�  �          A��Q�@n�R���\�p�C0���Q�@����{��
=C.                                    BxpI L  �          A���G�@tz���{���C����G�@��R�r�\���C�                                    BxpI�  �          AG����@Z�H��33�Q�C�����@�����G����HC��                                    BxpI�  �          A���޸R@�
=���H���C�޸R@�p��E��ffC!H                                    BxpI,>  �          Az���ff@s33����\)C}q��ff@�  �|����33C�)                                    BxpI:�  �          A���
=@�����=q��\)C����
=@�=q�   �\)CO\                                    BxpII�  �          A���(�@�  �����=qCY���(�@�Q��=q�{33C	�                                    BxpIX0  �          A���{@�{��Q���p�C���{@�
=�"�\��33C�                                    BxpIf�  �          Ap���p�@Mp���Q���CL���p�@�Q��c33���\C�f                                    BxpIu|  �          A33����?�ff��ff�:  C*�����@>�R��=q�#p�C�R                                    BxpI�"  
�          A����@7���p�����C����@����E�����C�3                                    BxpI��  �          A=q��@�=q��G���G�C���@�{�5���33C{                                    BxpI�n  �          A����=q@��\�>�R��C����=q@�Q쿬���	�C�                                    BxpI�  �          A���G�@��&ff����C����G�@�\)�aG���Q�C޸                                    BxpI��  �          A�����@�p���  �4��C�����@�=�Q�?z�C
T{                                    BxpI�`  �          A��\@�(��\���C� �\@�p�?ٙ�A-Ck�                                    BxpI�  �          A(���p�@�p���G��/\)C��p�@���>��
?��RC                                    BxpI�  �          A\)��{@��ÿ��@Q�C�3��{@�=q>�?Tz�Cn                                    BxpI�R  �          A�
�ҏ\@�녿��R����C.�ҏ\@�z�?G�@�z�Cٚ                                    BxpJ�  �          A(���z�@�p��E����C����z�@ʏ\?��
@�p�CT{                                    BxpJ�  �          A��ҏ\@�p����H�Dz�C���ҏ\@�  ?��A�CxR                                    BxpJ%D  �          A���(�@�
=?   @J=qCW
��(�@�\)@%�A��
C��                                    BxpJ3�  T          A  ���R@�ff?n{@�G�C s3���R@�G�@H��A�33C#�                                    BxpJB�  �          A���\@�ff?0��@�Q�C+��\@�(�@5A��RC��                                    BxpJQ6  �          A(���z�@��?8Q�@�{C�{��z�@�33@1�A���C^�                                    BxpJ_�  �          A(����@���?�{@�(�B�.���@��@VffA�C��                                    BxpJn�  �          A\)���@��
>�(�@-p�B�\)���@ۅ@/\)A��RC ��                                    BxpJ}(  �          A�H���@�(�?&ff@��\B�p����@�G�@<��A�ffC ��                                    BxpJ��  �          A����(�@�G�?��@��C ����(�@ʏ\@QG�A��RC��                                    BxpJ�t  �          Ap�����@�G�>�@HQ�B��f����@У�@,(�A�=qC�                                    BxpJ�  �          A����z�@޸R?Q�@��HB�z���z�@��H@@  A�z�C^�                                    BxpJ��  �          A
=��p�@޸R?��RA (�B��)��p�@�ff@Y��A�G�C+�                                    BxpJ�f  �          A
=����@���>�{@  B�=q����@��@.�RA��RB�3                                    BxpJ�  �          A
{��33@�\?�G�A=qB�W
��33@�G�@b�\A��B�33                                    BxpJ�  �          A	����(�@�\?��HA8(�B�����(�@���@~�RA�z�B�                                    BxpJ�X  T          A���~{@�33?�Q�A6�RB����~{@�p�@}p�A�\)B�=                                    BxpK �  �          A	p���ff@�z�?��HAS�B��\��ff@�z�@�=qA�
=C �\                                    BxpK�  �          A
=q����@�\)@p�Am�B�=q����@�z�@��HA�  B�aH                                    BxpKJ  �          A�����@�  @�HA��
B������@�33@�
=A���Ck�                                    BxpK,�  �          Aff��{@ə�@c�
A�\)B�8R��{@��@���B!
=C�                                    BxpK;�  �          A���z�@��@�
=B�B�����z�@~�R@��HB:�RC#�                                    BxpKJ<  �          A���|��@��@c�
Aʏ\B�W
�|��@���@�ffB#z�C ޸                                    BxpKX�  �          A�R�`  @�@��A���B�\�`  @�=q@�(�B��B��                                    BxpKg�  �          A	G��l(�@���?�(�A9p�B�B��l(�@��@��\A�B�                                    BxpKv.  �          A	���p�@���?�A\)B��)��p�@���@p  AθRB�(�                                    BxpK��  �          A���u�@�Q�?�  A!B�  �u�@Ӆ@xQ�A�\)B�(�                                    BxpK�z  T          A\)�`��@���?��HAz�B����`��@�
=@g
=A�B�(�                                    BxpK�   �          A���o\)@�G�@s�
A��HB� �o\)@�@�  B+�B�ff                                    BxpK��  �          A���I��@B�\@�p�Bjp�C��I��?�\@���B�\)C*Ǯ                                    BxpK�l  �          A	p��P��@L��@�RBf�\C� �P��?&ff@��B�.C(�                                     BxpK�  �          Az��\(�@�
=@��B*(�B��q�\(�@A�@�z�Be�C
�\                                    BxpKܸ  �          A�\�b�\@�33@��\B'�HB�z��b�\@>{@�ffBa�C                                      BxpK�^  �          A�
�g�@�ff@���B.\)B�33�g�@$z�@�z�Bd�
C��                                    BxpK�  �          AQ���@�  @�G�Bz�C����@HQ�@�BB��CǮ                                    BxpL�  �          A(��!�@��R@��BPp�B�  �!�?�\)@�B���C}q                                    BxpLP  �          A�H�Y��@G
=@�\)B�ffBҙ��Y��>�G�Ap�B��C�R                                    BxpL%�  �          A녿�33@P  @޸RB~�\B�𤿓33?=p�@���B��3C)                                    BxpL4�  �          Aff�C33@h��@�G�BX�C�3�C33?��R@�(�B��C�\                                    BxpLCB  �          @�\)�J�H@���@���BB�J�H@s�
@���BL�C�=                                    BxpLQ�  �          A���L(�@��R@���B(�B�G��L(�@�z�@�z�BFffB�.                                    BxpL`�  �          A��fff@�G�@z=qA��B�fff@��@�G�B2�HC                                     BxpLo4  �          A�H�P��@�ff@Q�A�=qB����P��@��R@�33B$=qB�
=                                    BxpL}�  �          Ap��K�@���@�33A��
B��K�@�p�@�33B@�\B��3                                    BxpL��  �          A�
��=q@���?��A{C�)��=q@���@K�A���C	��                                    BxpL�&  �          Ap����@�G�@�As�C k����@�@�A�C�                                    BxpL��  �          A33�(�@��
@�  B-  B�ff�(�@@�(�Bk�C.                                    BxpL�r  �          A��W
=?�@�B���BȸR�W
=����@��B��
C�/\                                    BxpL�  �          A=q��
@Mp�@�G�Bo�B�ff��
?5@�\)B�33C"�                                    BxpLվ  �          A����z�@�Q�@G
=A��B�Ǯ��z�@��@�z�B��C                                      BxpL�d  �          Ap���ff@�Q�@.�RA�ffB�u���ff@�p�@�(�B\)Ck�                                    BxpL�
  �          A��\)@��H?�(�AO\)C�q��\)@���@��A���C�\                                    BxpM�  �          A�H��=q@��
?�(�A�C�H��=q@�\)@c�
A�Q�C	�                                    BxpMV  T          A�H���H@�p�?^�R@�z�C}q���H@�\)@A�A�33C�                                     BxpM�  �          A\)��{@���\��HCG���{@��?�\A6�RC	s3                                    BxpM-�  �          A  ��=q@�녿����Hz�C
����=q@��
>.{?�{C	                                      BxpM<H  �          A����ff@�=q�dz�����C����ff@θR���
=C�f                                    BxpMJ�  �          A����z�@�33��Q���C�f��z�@Å�#33��G�C�                                    BxpMY�  �          A�����R@��
��33�י�B��)���R@�z����'33B��R                                    BxpMh:  �          A����
=@����ff��
=C����
=@�{�(��{
=B��                                    BxpMv�  �          Az�����@�=q����\)C�)����@�\)�4z���  C 8R                                    BxpM��  �          A���p�@����33��z�CǮ��p�@�=q�
=�YC z�                                    BxpM�,  �          A�
��ff@�����\��
=C� ��ff@�  ���t��C ��                                    BxpM��  �          A
=��Q�@���
=�ffC
T{��Q�@ə��_\)��=qC�)                                    BxpM�x  �          A(�����@��w
=��(�Cn����@��У��'\)C��                                    BxpM�  �          A�R��33@�(����\(�C��33@�?�z�A,��C�q                                    BxpM��  
(          A�H�أ�@�����R�N�RC\)�أ�@���=#�
>�=qC�{                                    BxpM�j  �          A���\@�33��G��9�C���\@ʏ\?   @P  C�{                                    BxpM�  �          AQ���Q�@��׿������C���Q�@�Q�?�\)@��HC&f                                    BxpM��  �          A�
��@�(�����Dz�C���@�(�?���A@z�C\                                    BxpN	\  �          A�
��=q@�z῕���B��3��=q@ۅ?��A33B��                                    BxpN  �          A�
���@�
=�33�w�B�Q����@�33>k�?��
B�W
                                    BxpN&�  �          A�
���\@�33��(��3�
B�Ǯ���\@љ�@�\AX��C�                                    BxpN5N  �          A���\)@˅>�\)?�\)C
=��\)@��H@!G�A��Cu�                                    BxpNC�  �          A�����
@У׿�
=��
=Cs3���
@�Q�?�(�A�
Cz�                                    BxpNR�  �          A���33@��
�}p���G�C�=��33@��?���A�HC�
                                    BxpNa@  �          A����@�\)>u?�=qB�Ǯ���@���@9��A���B�                                      BxpNo�  �          A�
���@�33>��@.�RB����@�\)@;�A�
=B��R                                    BxpN~�  �          A(���(�@�=q���E�C J=��(�@��@Az=qC
=                                    BxpN�2  �          A����
=@陚����!�B��q��
=@��
?�Q�@��B�B�                                    BxpN��  �          Ap����@������\B������@�?�ffA
=B���                                    BxpN�~  �          Ap�����@�
=��  ��\Cff����@ָR?�G�AQ�Ch�                                    BxpN�$  �          A�\��
=@�  �s33��(�C޸��
=@�p�?���A33C=q                                    BxpN��  �          A=q��
=@�zῦff�\)Cff��
=@�{?�G�@ҏ\C&f                                    BxpN�p  �          A=q��\)@����
=��
C	����\)@���?Q�@���C	
                                    BxpN�  �          A=q��
=@��Ϳ���,  C33��
=@�(�>�G�@8Q�C\                                    BxpN�  �          A����
=@�z��  �8z�CE��
=@��R=u>���C��                                    BxpOb  �          AG����@�\)�k���Q�B�aH���@�=q?��
A;33B���                                    BxpO  �          AQ���{@������}p�C����{@���u�˅C	u�                                    BxpO�  �          A=q�ə�@��H����RC���ə�@�{?k�@��C��                                    BxpO.T  �          Aff���H@У׿˅�$z�C�)���H@�z�?k�@�p�CL�                                    BxpO<�  T          A�R�˅@����=q�$  C� �˅@���?:�H@�\)C�q                                    BxpOK�  �          A��ۅ@�녾W
=��{C  �ۅ@��?���AA��C��                                    BxpOZF  �          Ap���G�@׮?
=q@aG�Ck���G�@�=q@?\)A�Cff                                    BxpOh�  �          A����\@�?�@aG�C�)���\@�Q�@>{A��C޸                                    BxpOw�  �          A����
=@��H>��R@G�C �H��
=@Ǯ@4z�A�(�C8R                                    BxpO�8  �          A{��z�@�
=�6ff��z�B�\)��z�@��ýu���B�Q�                                    BxpO��  �          A��Q�@�z��e���(�B�W
��Q�@�  �Y����=qB�\                                    BxpO��  �          A=q���@�  ���R�G�C\)���@�(�@{Ahz�C��                                    BxpO�*  �          A=q���@ڏ\        Cff���@��H@#33A�  C�                                    BxpO��  �          A�\��p�@�33��t(�B�\��p�@�\)>Ǯ@   B��                                    BxpO�v  �          A����@�
=��{��Cn����@�
=��p��O33B��)                                    BxpO�  �          AQ����@�(��33�{�B��H���@�Q�>�33@ffB��)                                    BxpO��  �          A����@�����o
=B�B����@�p�>�ff@8Q�B�u�                                    BxpO�h  T          A������@Ϯ��
�Y��C^�����@�G�?�@^�RC�                                    BxpP
  T          A����{@�\)��  �(�CJ=��{@�
=?��
A\)CO\                                    BxpP�  �          Ap�����@�33��(��5�C �����@�Q�@�Aep�C^�                                    BxpP'Z  �          Aff���
@�G��
=�r�\B��3���
@׮@�A\��C ��                                    BxpP6   �          A�R���@У���s33C�q���@�p�>��R?��HC\                                    BxpPD�  �          A�R�˅@�\)�'����C�{�˅@��\��G��:=qC	}q                                    BxpPSL  �          A�\��ff@l(���  �ׅC���ff@�{�33�q�C�)                                    BxpPa�  �          A���׮@Vff�u��  C���׮@���G��yp�C�                                    BxpPp�  �          A������@�G�����  C	�)����@�33?��\@�{C	�
                                    BxpP>  �          A������@޸R?
=q@a�B�aH����@�
=@J=qA�33CL�                                    BxpP��  �          A\)����@���?�(�A�\C �\����@�@j�HA�ffC
=                                    BxpP��  �          A���=q@��H@'
=A��RB��=��=q@�=q@���B��C�3                                    BxpP�0  �          A���
=@ָR?�33A/�
B�����
=@��@�33A�{C5�                                    BxpP��  T          A���
=@�=q�c�
��z�C����
=@�Q쿑����
C��                                    BxpP�|  �          A����@�ff�Fff���C0�����@��\)�i��C޸                                    BxpP�"  �          A�
���\@�z��^�R��p�C&f���\@��ÿp����
=C .                                    BxpP��  �          A
�H���\@�G��!����B������\@�Q�>L��?�ffB���                                    BxpP�n  �          A
�R��=q@�\)?��@�{B�����=q@�Q�@i��A�CT{                                    BxpQ  �          A	���@أ�?��RAffB������@�Q�@p��AӅCz�                                    BxpQ�  �          A	����@ə��xQ��׮C����@�?�ffA,Q�C�f                                    BxpQ `  �          A	������@�Q���R�\��B�������@أ�?333@��B�z�                                    BxpQ/  �          A
ff���
@ۅ���H�7�B��R���
@�\)?���@�z�B�                                    BxpQ=�  �          A33��G�@��H���H�p�B�L���G�@�33?�A��B�B�                                    BxpQLR  �          A���Q�@�G��\)�n{B�����Q�@߮@4z�A�ffB��                                    BxpQZ�  �          A���G
=@��R@z�A��HB�\�G
=@�p�@��BG�B�=q                                    BxpQi�  �          AQ����@�ff@�=qB{CL����@�R@�\)BM�C��                                    BxpQxD  �          A(�����@�Q�@�Q�B��C� ����?���@�Q�B@�C!��                                    BxpQ��  �          A����@��@�p�B�\CQ����@(�@�(�B4�C�                                     BxpQ��  �          A
=�l(�@��@��B-
=B�u��l(�@��@�{Bl��C:�                                    BxpQ�6  �          A
�\�C33@�33@�\)BS=qB�ff�C33?���@���B���C &f                                    BxpQ��  �          A�
�J=q@�\)@�  BCG�B�z��J=q?�\@��HB�k�C�q                                    BxpQ��  �          A����(�@�  @[�A�G�B����(�@��H@�=qB)�
Cz�                                    BxpQ�(  �          AQ���
=@�\?�z�AG�B�Ǯ��
=@�p�@��RA�G�B��f                                    BxpQ��  �          A
�H��p�@陚>\@!G�B�33��p�@�G�@P  A�=qB�                                      BxpQ�t  �          A	�����@ָR?L��@��B�{����@��\@Z=qA�33CxR                                    BxpQ�  �          A\)��{@���@.{A��C
E��{@~{@�ffB\)CJ=                                    BxpR
�  �          Ap��أ�@���?��AH��C���أ�@}p�@s33AΏ\C��                                    BxpRf  �          A�����@���>B�\?�(�C�q���@���@�\Aq��CǮ                                    BxpR(  �          A  ��@��׿\(���(�C����@�?�=qA�C@                                     BxpR6�  �          A(����@�33��{�+
=C=q���@���?+�@�C:�                                    BxpREX  �          A{��G�@��R@J=qA���C!H��G�@�{@�G�B(�C�                                    BxpRS�  �          A�\���@�@�AZ=qC
s3���@��@�{A�33C                                    BxpRb�  �          A���
=@�{��G��@  C���
=@���@AZ�RC=q                                    BxpRqJ  �          A�Ӆ@�{������CJ=�Ӆ@���?��
@�ffC
�                                    BxpR�  �          A�����
@�=q��z��G
=C�3���
@��?�\@R�\C
z�                                    BxpR��  �          A���˅@�z��0  ��  C� �˅@�G��������C�                                     BxpR�<  �          A{��(�@���J�H��p�C	�=��(�@��
�!G����HC�                                    BxpR��  T          A����=q@�(��?\)��Q�C����=q@�(��
=q�`��C�H                                    BxpR��  T          A���љ�@�
=�Y������C0��љ�@�ff��
=��  C
��                                    BxpR�.  �          A���Q�@}p������
=C���Q�@�����h��C�                                    BxpR��  �          A����p�@�(��(Q���G�C���p�@���L�Ϳ��
C�q                                    BxpR�z  �          A��Q�@��\�(��eG�C^���Q�@��>�z�?�\)C	^�                                    BxpR�   �          AG���=q@��
�"�\��  C
W
��=q@�p����
����C�                                    BxpS�  �          Ap���ff@����/\)����C	����ff@��þ8Q쿑�C��                                    BxpSl  �          AG���
=@�z��L����C���
=@�\)�0����  C�3                                    BxpS!  �          A(���  @�{�3�
��ffC����  @��H�aG���
=Cc�                                    BxpS/�  �          A����(�@����.{���\C	W
��(�@ȣ׾���s33C\)                                    BxpS>^  �          A����
=@����	���aC�=��
=@�Q�>�@EC��                                    BxpSM  �          AG���Q�@�����R�j�\C���Q�@��>��@C33C+�                                    BxpS[�  �          AG���=q@����\�p(�Cc���=q@�z�>���@'�C�                                     BxpSjP  �          A����@�p���Q��3
=C�)��@�=q?��\@׮C0�                                    BxpSx�  �          A����G�@У׾��H�N{C����G�@�z�@G�Ap��C^�                                    BxpS��  �          A������@�(�>�\)?�=qC�����@��@AG�A��\CO\                                    BxpS�B  
�          AG���=q@�33?
=q@a�C^���=q@���@P  A�
=C33                                    BxpS��  �          A���\)@У׽��
��\C�\��\)@�ff@)��A��C&f                                    BxpS��  �          A�����@У׿!G����C�����@�@
=qAc33C}q                                    BxpS�4  �          AQ���{@�{��Q���C����{@�z�@'�A�
=C@                                     BxpS��  �          A(����\@�
=�}p��θRC����\@�G�?���A@  C�{                                    BxpS߀  �          A�R��33@��
������z�C���33@�  ?�
=A-�CW
                                    BxpS�&  �          A=q�׮@�\)��G��7�C�)�׮@�(�@ ��AR{C^�                                    BxpS��  �          A
=��ff@��
�@  ��=qC
���ff@�z�?�ffA9G�C��                                    BxpTr  �          A33�θR@���
=q�]p�CW
�θR@��@
=AZ�HC
                                      BxpT  �          A�\��
=@�(�=�G�?.{CG���
=@�\)@2�\A�=qC	T{                                    BxpT(�  �          A=q��
=@��=u>\CO\��
=@�p�@5�A���CB�                                    BxpT7d  �          A\)��=q@���?��@`��C�H��=q@��@W�A��C�\                                    BxpTF
  T          A�H���@�Q�?�G�@�\)CQ����@�{@q�AȸRC:�                                    BxpTT�  �          Ap���@���=u>�p�C8R��@�z�@5�A�C33                                    BxpTcV  �          A���ȣ�@�
=�Ǯ�   C8R�ȣ�@�G�@33Aq��C	E                                    BxpTq�  �          Ap���Q�@�p�=�?E�C���Q�@�  @6ffA�
=C@                                     BxpT��  �          A������@˅?z�@s33CxR����@�  @O\)A�p�C	�H                                    BxpT�H  �          A����@Ϯ?�@[�Cu���@�z�@P  A�p�Cz�                                    BxpT��  �          AG�����@�p�?��@���CB�����@�=q@tz�A�p�Cc�                                    BxpT��  �          AG����\@�z�?���A\)C����\@�(�@��A�C{                                    BxpT�:  �          AQ���
=@��?z�H@��C���
=@�  @l��A�{C&f                                    BxpT��  �          A\)��@���?˅A*{C�=��@�Q�@���A�\)CE                                    BxpT؆  �          A�����@ۅ?���A�HB�������@��@�p�A��
C��                                    BxpT�,  �          A���ff@�  ?��A
�\B�� ��ff@�  @�ffA��
C��                                    BxpT��  T          A
�\�5�@�@%�A�  B�k��5�@��@�z�B$��B��f                                    BxpUx  �          A
�H����@�?O\)@�z�C �����@�p�@hQ�A���C�{                                    BxpU  T          A
�H��Q�@�����\��  C�H��Q�@���?�33A/
=C	T{                                    BxpU!�  �          A	���Q�@���������C	5���Q�@��R?�  A z�C	��                                    BxpU0j  �          A	���G�@�zῠ  �C� ��G�@�z�?�(�A\)Cz�                                    BxpU?  �          A\)���@�Q쾸Q��{C�
���@���@��A��RC\                                    BxpUM�  �          A�����@�p���z���RC	�����@�\)@p�Aq�C8R                                    BxpU\\  �          A�H���@���?:�H@�  C �=���@���@`��A�p�CaH                                    BxpUk  �          A�����\@����u�\C�R���\@��R@&ffA��C�H                                    BxpUy�  �          A
�H���
@Ǯ��=q���C8R���
@\?�\A=�C�R                                    BxpU�N  �          A
�R����@���\(����C33����@���?�  A33C�q                                    BxpU��  �          Az����H@��:�H��  C�\���H@�{?�p�A6�\CǮ                                    BxpU��  �          A�H��ff@��>�\)?�C���ff@�33@3�
A��
C�q                                    BxpU�@  �          A
=q��z�@���@;�A���C33��z�@^{@�
=B�\C�=                                    BxpU��  T          Az���z�@��@33As�
C����z�@�z�@�33BQ�C�\                                    BxpUь  T          A����(�@��
@=p�A���C�)��(�@�@��B�\Cff                                    BxpU�2  �          A�����@��\@c�
A��
C�{���@[�@��B!  C+�                                    BxpU��  �          A{���@��H@�=qA�33C�=���@Fff@�33B6�C��                                    BxpU�~  �          A�H���\@��@�B�HC �����\@�\@�
=Bb�HC0�                                    BxpV$  �          Aff���@�Q�@�\)B p�C�����?�(�@�  BZQ�C ��                                    BxpV�  �          A�����@�z�@�p�B.z�C8R����?�z�@��HBh=qC"��                                    BxpV)p  �          A��~{@�@ϮB:p�CW
�~{?�ff@���Bxp�C!�                                    BxpV8  �          A�����@��@�
=B  C ޸���@��@��B_{C0�                                    BxpVF�  �          A����
@���@��B"z�B��
���
@
=q@�  Bh��CL�                                    BxpVUb  �          A�\��p�@�׿J=q���\CL���p�@�(�@�Ap��C��                                    BxpVd  �          A�H��p�@�p����9��C T{��p�@��
@333A�Q�C�{                                    BxpVr�  �          A��=q@�\)���g
=C����=q@Ϯ@'�A��Cٚ                                    BxpV�T  
�          AQ���{@�Q��  �\Q�CJ=��{@���?��@�z�C=q                                    BxpV��  �          A����=q@�������p�C����=q@��Ϳ��H���HC .                                    BxpV��  �          A����Q�@��R��(���C}q��Q�@������  C�                                    BxpV�F  �          A������@�  ��z���(�C������@�\��\�/
=B�                                      BxpV��  �          A����@�z���,(�C {���@��@B�\A��Cs3                                    BxpVʒ  �          A��Q�@�ff�@  ���RC ����Q�@߮@,��A���C��                                    BxpV�8  �          A33��p�@�33�.{��z�C �{��p�@ۅ@.{A���C                                    BxpV��  �          Az����@��
?^�R@�C�����@�Q�@q�A�{C
��                                    BxpV��  �          A����@����ff�1�C����@��
@/\)A�\)C
=                                    BxpW*  �          Ap��˅@���?z�@c33C0��˅@���@eA��C	�                                    BxpW�  �          A����  @�p�?�
=A>�RC+���  @��@�  A�\C�                                    BxpW"v  �          A�H��Q�@��@\)A}G�C0���Q�@�=q@�z�B�HCǮ                                    BxpW1  �          AQ���Q�@�����
�\C���Q�@ҏ\?�A4  C��                                    BxpW?�  T          A���
=@��
�=p���G�C(���
=@�ff@�RAw�
C��                                    BxpWNh  �          Az���\)@��ÿk���C�{��\)@�ff@�\Ad��C                                    BxpW]  �          A(���{@�ff���\��z�C T{��{@�\)@
=qAX(�C.                                    BxpWk�  �          A  ��  @����\)�  C(���  @љ�?�A8  C��                                    BxpWzZ  �          A  ����@˅���R�G33C&f����@�=q?�33@���C=q                                    BxpW�   �          AQ����
@�33�{�]C޸���
@�(�?��@��C�                                    BxpW��  �          AG����@�=q�'����RC	:����@�33>��@7�Cٚ                                    BxpW�L  �          AG���z�@���c�
��C&f��z�@�
=�G���=qC	�f                                    BxpW��  �          Ap���G�@���6ff��(�C����G�@���=�?:�HC��                                    BxpWØ  �          A=q��33@�ff�(��p��C
=��33@���?z�@c�
C�                                    BxpW�>  �          AG��ٙ�@��
��Q��{C�H�ٙ�@��@'�A���CxR                                    BxpW��  T          A\)��@љ����R���HCs3��@�ff@0  A���C	0�                                    BxpW�  T          AQ����H@Ǯ?ٙ�A0(�CT{���H@��@�z�A�33C�                                    BxpW�0  �          A=q��
=@���?�G�@�z�CY���
=@�G�@�G�A�G�C	#�                                    BxpX�  �          A���z�@�G�?Y��@���C)��z�@��
@z=qA�33C	��                                    BxpX|  �          A����
=@�Q�=�\)>�
=C	���
=@�Q�@>{A���C��                                    BxpX*"  �          AQ���G�@���?��@�z�C����G�@��@w�A�33C                                      BxpX8�  �          A��=q@ƸR@~�RA�\)C�{��=q@h��@�p�B3ffC�)                                    BxpXGn  �          A=q����@ҏ\@��
A���B�������@y��@�
=B=��C��                                    BxpXV  �          Ap����@�{@G
=A���C�����@��@�Q�B
=C�                                    BxpXd�  �          A���33@�z�?=p�@�ffCY���33@��@xQ�Aʣ�C��                                    BxpXs`  �          A�R����@أ�>�=q?�CJ=����@�33@Y��A���Cp�                                    BxpX�  �          A{��
=@���>�\)?��
C���
=@��@W�A��C	G�                                    BxpX��  �          Ap���G�@�  ���:=qC�{��G�@�p�@0��A��CaH                                    BxpX�R  T          A����(�@�����W
=C	���(�@�ff@7�A�C�{                                    BxpX��  �          Ap��ٙ�@�ff��ff�7
=C
���ٙ�@��R@��AvffC:�                                    BxpX��  �          A��ڏ\@��H�fff��{C}q�ڏ\@�=q?��HAH��C�\                                    BxpX�D  �          AQ���Q�@�{��ff�6ffC5���Q�@�z�@(��A��RC                                    BxpX��  �          A(���\)@�{�
=q�^{C����\)@���@+�A��RC0�                                    BxpX�  �          A����
@��
����&ffC����
@�  @8��A��HC8R                                    BxpX�6  �          A  ��ff@�p��G���  C��ff@Ǯ@{A�C�H                                    BxpY�  �          A����33@�G����\�G�CE��33@�z�?�{A>�RC��                                    BxpY�  �          A(��\@�=q�B�\��=qC��\@�z�@��A}��C�3                                    BxpY#(  �          A�
��p�@Ϯ�5����C����p�@���@p�A~{C��                                    BxpY1�  �          A  ��  @�z���Y��C^���  @��
@$z�A��C�=                                    BxpY@t  �          A���ʏ\@�z�8Q�����C�R�ʏ\@��R@=qAx(�C�R                                    BxpYO  �          A�����
@�=q����7�C\)���
@�{?���A�C�\                                    BxpY]�  �          A����=q@�  �Q��s�
C�H��=q@��H?�G�@�p�Cff                                    BxpYlf  �          AG���ff@�녿�\�N�RC
��ff@�33@�Am��Cff                                    BxpY{  �          Ap���(�@�ff��ff�6�HC
\��(�@��H?�G�A   C	h�                                    BxpY��  �          A����@���Q��X  C	xR����@���?�  @ə�C!H                                    BxpY�X  �          A���Ϯ@��׿��<  C	&f�Ϯ@�p�?��\A�CxR                                    BxpY��  �          AG��љ�@��Ϳ�����\C���љ�@�
=?�33AAC	�f                                    BxpY��  �          A���  @���ٙ��,z�C
=��  @��R?���A��C
��                                    BxpY�J  �          A{�ٙ�@�ff�(������CO\�ٙ�@���>���?�C
^�                                    BxpY��  �          A��G�@�ff�;�����C���G�@�  �B�\���HCǮ                                    BxpY�  �          A=q��@�\)�7
=���
C�R��@����G��(��C\                                    BxpY�<  �          A{��z�@�\)�8����CǮ��z�@�\)�#�
�L��C\                                    BxpY��  �          A{���
@�����N{CB����
@�(�?J=q@�ffC��                                    BxpZ�  �          A���  @�ff��S�C)��  @���?O\)@��\CxR                                    BxpZ.  T          A����@�ff���R�I��C0����@�\)?fff@���CǮ                                    BxpZ*�  �          A���@��׿�\)���C����@��?�(�A\)C��                                    BxpZ9z  �          AQ���Q�@����ff��C���Q�@���\)���C �                                     BxpZH   �          A�
����@�����z����C�{����@�녿Q����\C�)                                    BxpZV�  �          A�R��p�@�z���\)��C	����p�@�zῥ��C��                                    BxpZel  �          A����p�@�Q���=q���C	\��p�@�G��aG����\C�                                    BxpZt  �          AG���(�@������\)C	�
��(�@�z��ff�6�RCB�                                    BxpZ��  �          AG���
=@�������(�C	\)��
=@�(��\��C�                                     BxpZ�^  �          A������@�=q��33��C	�R����@�Q쿙����(�C)                                    BxpZ�  �          A{��=q@�Q��g
=����C}q��=q@�G���{��C�=                                    BxpZ��  �          A����@���dz�����C������@�=q��\)�޸RCn                                    BxpZ�P  �          A�����
@���z�H��\)C����
@�33�&ff��(�C��                                    BxpZ��  �          A����@�ff�������C
ff���@�{����(�C5�                                    BxpZڜ  �          A�����@��H�����
=C
s3���@�=q������C!H                                    BxpZ�B  �          Aff��G�@��\��{�{C����G�@���33�V{C\                                    BxpZ��  �          Aff��z�@�
=������RCu���z�@ȣ׿�G��z�Ch�                                    Bxp[�  �          Ap���(�@�(���=q��\)C����(�@�  �����{Cz�                                    Bxp[4  �          A���
@�(���������C  ���
@�Q쿝p����C��                                    Bxp[#�  �          A\)���H@��������C�����H@�  �s33�ÅC#�                                    Bxp[2�  �          A
=��Q�@������\���C���Q�@��ÿ������C�                                    Bxp[A&  T          A�׮@�\)����Q�CY��׮@��R����C
�                                     Bxp[O�  �          A
=�ۅ@�Q���33�ڣ�C���ۅ@�Q�˅�$  C8R                                    Bxp[^r  �          A���33@���S�
���
B��R��33@��H>�Q�@�B��{                                    Bxp[m  �          A���33@θR�K���33C G���33@�{>�(�@2�\B��                                    Bxp[{�  T          A�����@���Mp����C�����@�
=>\)?k�Cc�                                    Bxp[�d  �          A	����@���Q����C������@ʏ\�.{���C�                                     Bxp[�
  �          A
=���
@�����L��C  ���
@���@:=qA��
C�                                    Bxp[��  �          A	p����@�p�?��@陚C�H���@�G�@�(�A��HC	�
                                    Bxp[�V  �          A����
=@�  ?^�R@�33C	E��
=@��@g
=Aȣ�C�=                                    Bxp[��  �          A
=��=q@�ff?��HA�C8R��=q@�(�@p��A�C�
                                    Bxp[Ӣ  T          A�\��G�@�\)?�33@�(�C
��G�@��@~�RA��CxR                                    Bxp[�H  �          A\)����@���?��\A	�CQ�����@��@�\)A�  C
�                                    Bxp[��  T          A\)����@��?�33AQ�B������@�G�@��B�
CT{                                    Bxp[��  �          A��W�@���?�z�AP��B�.�W�@�=q@�p�B
=B�L�                                    Bxp\:  �          A���Y��@��@   AY�B�ff�Y��@���@�Q�B!
=B���                                    Bxp\�  T          A33���\@ۅ@*=qA��B�=���\@��@�\)B+�
C��                                    Bxp\+�  T          A����@�33�|����Ck����@�ff��\)���C
=                                    Bxp\:,  �          A �����@�{�o\)���C�H���@�
=��G���G�Cp�                                    Bxp\H�  �          Aff��
=@�������Q�C���
=@�p�?�33AX��C�                                    Bxp\Wx  �          A(���  @�{@z�A��HC����  @p��@��
B��C��                                    Bxp\f  �          A�R��@���?s33@�\)C	�3��@�{@a�A�G�C�                                     Bxp\t�  �          A�H��=q@��׾�  �޸RC	����=q@�z�@#�
A�\)C�                                    Bxp\�j  �          A ���N�R@�{@e�A�33B�W
�N�R@qG�@ʏ\BO��C��                                    Bxp\�  �          A z��vff@�\)@*�HA�B�u��vff@�Q�@�=qB0p�C#�                                    Bxp\��  T          A Q�����@�@z�Ao33B��=����@�Q�@���Bz�C��                                    Bxp\�\  "          @�
=���@������c�
C0����@�G�@<��A��\C�                                    Bxp\�  �          @����{�@���?�p�AG�B�#��{�@���@�G�B	ffC �q                                    Bxp\̨  �          @�
=����@��
    �#�
C\����@��@5A��HC@                                     Bxp\�N  �          @��R��(�@�
=���R�3�
C
�{��(�@�G�?�(�A33C
+�                                    Bxp\��  �          @�\)��=q@�  ��ff�:�\C�\��=q@�(�?���@�\)C
=                                    Bxp\��  �          @����z�@�����\����C����z�@�z�=u>�C                                    Bxp]@  �          @������
@�녿����B�HCxR���
@�ff?��A ��C
��                                    Bxp]�  �          @�  ���@�\)?(��@���C�\���@�33@^�RAڸRC��                                    Bxp]$�  �          @����@  @���@S33A�B�
=�@  @{@���BV��C��                                    Bxp]32  �          @��
�k�@�G�@�G�BIz�B���k�?�
=@�p�B��B�\                                    Bxp]A�  T          @��Ϳ�\)@��@\BT33B�𤿯\)?+�@�ffB��)C)                                    Bxp]P~  �          @��H�#�
@�G�@��
B@\)B����#�
?�p�@��B�=qB�8R                                    Bxp]_$  �          @�(��W�@��@u�A��RB�p��W�@#�
@��RBZ��C�                                     Bxp]m�  �          @�׿@  @dz�@��BW�B���@  ?�@ÅB���C��                                    Bxp]|p  �          @�(��=p�@�G�@��RBB{Bƞ��=p�?�  @ٙ�B�ǮB�k�                                    Bxp]�  �          @�33�Q�@�{@��HB-
=B��)�Q�?Ǯ@��B�(�CǮ                                    Bxp]��  �          @�G��$z�@��H@��B/B�=q�$z�?�  @�z�B��)C�                                    Bxp]�b  �          @�33����@�\)@#�
A���Ch�����@"�\@��B#�CǮ                                    Bxp]�  "          @�
=��  @��\��33���C����  @��?��RAG�
C�                                    Bxp]Ů  �          @�������@�33?��
Al��C�\����@5�@p��BffC@                                     Bxp]�T  �          @�=q���\@�
=@1G�A���C
�)���\@p�@�G�B"{Cs3                                    Bxp]��  �          @�p��	��@AG�@ÅBj�HB��f�	����@�G�B��
CA��                                    Bxp]�  �          @�\)�W�@�{@�  B%�
C � �W�?�z�@�\)Bp��C �                                    Bxp^ F  �          @���p  @��@�(�B�C
=�p  ?���@�z�B`  C �{                                    Bxp^�  �          @�=q��\)@��\@i��A�33C����\)?�@��B?��C�3                                    Bxp^�  �          @�=q���R@�(�@Z�HA�(�C	
���R@   @�{B5�\C�                                    Bxp^,8  �          @������
@�p�@{A�Q�C�����
@!G�@�z�B��C�)                                    Bxp^:�  �          @��H����@��H@
�HA��
C
�����@4z�@�
=B�C�
                                    Bxp^I�  �          @�(�����@�>�
=@S�
C	�{����@�Q�@9��A��RCc�                                    Bxp^X*  �          @�����33@�@�A��Cz���33@AG�@���B"�C=q                                    Bxp^f�  T          @������@�@#�
A�{C�����@:=q@�\)B�\C)                                    Bxp^uv  �          @�G���G�@��@l(�A�C�=��G�?�=q@�  BD  C��                                    Bxp^�  �          @�Q�?\)>8Q�@��B��A�
=?\)�W
=@�p�BpG�C��)                                    Bxp^��  �          @���@2�\�У�@�ffB�.C��
@2�\��\)@��B'=qC��R                                    Bxp^�h  �          @��R@S�
��\)@�  Bu�C���@S�
���
@�{B��C��                                    Bxp^�  �          @�(�@C33���@�\B��C��\@C33���@�Q�B(
=C���                                    Bxp^��  �          A ��@?\)�u@���B���C�q@?\)����@��\B:{C�aH                                    Bxp^�Z  �          A
=@333�Ǯ@�p�B��{C���@333���@˅BL33C�|)                                    Bxp^�   �          @��@\)>B�\@�=qB�z�@�33@\)�p��@�=qB^�C��                                     Bxp^�  �          A z�@�?�(�@��B���B\)@��*=q@�\)B��C��                                    Bxp^�L  �          @�(�?��?�33@�=qB���Bp�H?���33@���B�� C���                                    Bxp_�  �          @�33?O\)@��
@�  B:B�#�?O\)?\@�B��3Bx\)                                    Bxp_�  �          @���?��@Q�@�33Bs�B�z�?���   @�(�B�ǮC��R                                    Bxp_%>  T          @���?��
@.{@�B�� B��q?��
��33@��
B��fC��                                    Bxp_3�  �          A ��@8Q�?E�@���B�{An{@8Q��L��@�
=BeffC�f                                    Bxp_B�  �          Az�@u��Ǯ@��By{C�{@u���z�@�G�B;�HC�\)                                    Bxp_Q0  �          A�@U���  @�{B��C��q@U����H@�
=BF�HC��{                                    Bxp__�  �          A Q�?ٙ�?���@�\)B��B�H?ٙ��@��@�
=B��HC��
                                    Bxp_n|  �          A��?�?��@�Q�B�=qA�  ?��G�@�{B{��C�Ф                                    Bxp_}"  �          A�\?�  ?�=q@�ffB��=B#��?�  �N�R@�33B��\C���                                    Bxp_��  �          A?���?��@���B���B;=q?����I��@�\B�#�C�\)                                    Bxp_�n  �          A   ?�p�?ٙ�@�B�z�BX(�?�p��$z�@�{B��C��\                                    Bxp_�  �          @�>.{?8Q�@��B�=qB�u�>.{�_\)@�\B~��C�e                                    Bxp_��  �          @��>u?G�@�{B��B��\>u�W
=@޸RB�\)C��                                    Bxp_�`  �          A   @�?��R@��HB���A�@��:�H@�B}
=C��q                                    Bxp_�  �          @�\)@p�?��R@�G�B���A���@p��8��@�=qB{(�C��R                                    Bxp_�  �          @��
?��?���@�\B���B(�?���:�H@ٙ�B�k�C�H�                                    Bxp_�R  �          @�(�?�p�>k�@��B��A*�H?�p��hQ�@�33Bk\)C�^�                                    Bxp` �  �          @�  �3�
@Q�@�33B^��C�)�3�
�
=q@�(�B�33C>                                    Bxp`�  �          @�  �*�H@!G�@�=qBv�HC�H�*�H��
=@��
B���CPB�                                    Bxp`D  �          @�R��
@z�@��B��\C޸��
��@�(�B��C]W
                                    Bxp`,�  �          @�G�@G��L��@�
=B�(�C�AH@G����R@���BB�C��{                                    Bxp`;�  �          @��ÿ��R@#33@љ�B~�B��f���R���\@��B�G�CT��                                    Bxp`J6  7          @���g
=@�
=@��B-G�C���g
=?W
=@�p�Bs�
C&�3                                    Bxp`X�  T          @��R��{@��@�Q�B�HC޸��{?���@��HB^  C$                                      Bxp`g�  �          @�
=�z=q@�(�@�B)G�Ch��z=q?J=q@ҏ\Bk�C(��                                    Bxp`v(  �          @��H�K�@~�R@��
B?33C ���K�>�
=@��
B��HC,xR                                    Bxp`��  �          @�z��N�R@�p�@�\)B8G�B�z��N�R?(��@�33B���C(n                                    Bxp`�t  "          @�������@z=q@��\B-��C�R����?�\@��HBi  C,�                                    Bxp`�  
�          @��
�s33@�z�@�Q�B1C�=�s33?(�@ۅBr�C*ٚ                                    Bxp`��  �          @�G��j�H@���@��B.33C���j�H?L��@ٙ�Bt=qC'��                                    Bxp`�f  T          @������\@���@�  B ��C�H���\?�  @�  Bd��C&E                                    Bxp`�  
Z          @�33�~�R@�=q@��
B��C���~�R?��@�Q�B\ffC�)                                    Bxp`ܲ  
�          @�\��@'�@���B;C�H���\)@��
BY=qC;�                                    Bxp`�X  T          @�{�~�R@��@�\)B{C��~�R?���@�(�B^33C �f                                    Bxp`��  "          @��n�R@�
=@�Q�B=qC T{�n�R?�  @ÅB`
=Cٚ                                    Bxpa�  �          @��o\)@��@�z�B+C�3�o\)?8Q�@�  Bn�C)�                                    BxpaJ  �          @����R@�p�@���B  C�����R?��R@�
=BT�RC ��                                    Bxpa%�  
Z          @�G���
=@���@mp�A�Q�C+���
=?��@�G�B>  CJ=                                    Bxpa4�  
�          @���p�@���@�ffBffC�q��p�?��@��BS(�C#�R                                    BxpaC<  �          @������@��R@��RB�C������?�@���BV�C!�                                    BxpaQ�  �          @�{�xQ�@���@�33B(�C!H�xQ�?���@�Bc\)C��                                    Bxpa`�  �          @�������@~{@�{B��C�����?aG�@\BPp�C)T{                                    Bxpao.  "          @�(���@a�@���B��Ck���>��
@�z�BL�C0O\                                    Bxpa}�  �          @�Q����H@vff@*=qA�Q�C�����H?���@���B33C#!H                                    Bxpa�z  �          @��R����@���@ffA��\Cz�����@��@�ffB\)Ck�                                    Bxpa�   �          @����33@���?xQ�@�\)C�f��33@@  @A�A�  C޸                                    Bxpa��  
�          @�����
=@��\�33���C���
=@�>W
=?�  C
#�                                    Bxpa�l            @����\)@��\��R���C���\)@�33>�
=@L(�C                                    Bxpa�  �          A Q����
@�ff��p��dQ�C{���
@�=q?0��@�ffC��                                    Bxpaո  �          @�����@r�\�(Q���{Ck����@���W
=����C5�                                    Bxpa�^  �          A���@��
�QG���G�C�
��@�{��G��FffC	u�                                    Bxpa�  �          Aff��{@��\�B�\��\)C����{@��R        C�f                                    Bxpb�  T          Az���z�@��\�3�
��33C�3��z�@��H>L��?��C�                                    BxpbP  
�          A��Ӆ@�����H��ffCk��Ӆ@��
��ff��
C��                                    Bxpb�  
Z          A
�R�׮@�\)�Fff���C^��׮@�{�k��\C�                                    Bxpb-�  �          A ������@���������C�����@�(��
�H�yC {                                    Bxpb<B  T          A Q�����@g
=�����=�C
J=����@�ff�C�
���B�Q�                                    BxpbJ�  
�          @�ff�Z�H@.{���
�P�\C���Z�H@����S33��33B��q                                    BxpbY�  
�          @�{�s33?�=q��z��O�HCB��s33@��e���Q�C�H                                    Bxpbh4  "          @��H�!G�?�{���{CJ=�!G�@}p���{�'
=B�                                      Bxpbv�  �          @���X��?z�����u��C*@ �X��@w������1z�C5�                                    Bxpb��  
�          @��P��@���(��^G�C���P��@�z��l����=qB��                                    Bxpb�&  7          @�\)�^�R?W
=��  �p�RC&p��^�R@������\�(�\C�                                    Bxpb��  
�          A33���
?���Å�B�C..���
@n�R���
��C8R                                    Bxpb�r            AQ���zΎ������G�C?  ��z�?����G��C)c�                                    Bxpb�  7          A(���?�ff��=q�<33C$
��@��������C                                    Bxpbξ  
�          A�H�|��@���ָR�]�C(��|��@����33� ��B�Ǯ                                    Bxpb�d  i          A (�����@����G��[��CE����@����33��
B��                                    Bxpb�
  
�          @��R��p�?�ff��{�L=qC���p�@��\��Q���\C�q                                    Bxpb��  �          A (���  ?�
=���R�@(�C���  @�=q��  ��C�                                    Bxpc	V  
�          A��{?��R��(��D
=C\��{@�
=������C�)                                    Bxpc�  
�          A ����@�R��z��G33C�=��@�p��~�R���C&f                                    Bxpc&�  �          @�{��z�@����z��KQ�Cff��z�@���y�����C��                                    Bxpc5H  �          @�����@G����\�*�
C�����@�=q�Z�H��  C�
                                    BxpcC�  T          @�=q��ff@\)��  �3=qCE��ff@���S�
��ffCaH                                    BxpcR�  �          @�33����@'���=q�,Q�C������@�=q�E���ffC��                                    Bxpca:  "          @�p�����@&ff����
=C������@�z�������C��                                    Bxpco�  T          @��
���
@1��W���(�C�����
@��Ϳ����/
=C��                                    Bxpc~�  �          @�ff��@(Q��QG�����C!H��@~{�����:ffC��                                    Bxpc�,  T          @׮��p�@,(��W���=qC����p�@��\��33�?�C�                                     Bxpc��  �          @����
@>{��z���{C#����
@a논��
�#�
C�
                                    Bxpc�x  
�          @����@,(�����p�C�����@c33�����  C�                                    Bxpc�  
�          @���G�@����  ���C@ ��G�@����33���C��                                    Bxpc��  T          @�\��=q@�R��Q��)��C����=q@����+����\C��                                    Bxpc�j  "          @����R@p���{�#=qC����R@�(��1G���\)C
�=                                    Bxpc�  �          @�z���=q?�\)����@z�C"aH��=q@xQ��a���{C
�                                    Bxpc�  �          @�33��  ?�Q����R�5�RC�{��  @��Vff��{C��                                    Bxpd\  T          @�p���G�?�ff��{�J��C0���G�@��\�tz����RC�                                    Bxpd  T          @�{��\)@Mp���{�ffCB���\)@��H�G��|��Cc�                                    Bxpd�  �          @���Q�@%���p���CǮ��Q�@�{�#33���C��                                    Bxpd.N  
�          @�z���
=@~{����((�C@ ��
=@�G�?�ffAQ�C�                                    Bxpd<�  �          @����33@!G�@vffB z�C����33=#�
@�33B��C3�\                                    BxpdK�  �          @�R�Å@.{@R�\AхC��Å?�@�\)B
p�C/.                                    BxpdZ@  T          @����Q�@��@X��A�ffC)��Q�<�@�G�B(�C3�q                                    Bxpdh�  �          @�(����H@'��p���C�R���H@-p�?�R@���C��                                    Bxpdw�  
�          @������@*=q�P  ����C5�����@\)����2�RC�
                                    Bxpd�2  �          @��H��@`  ���:=qCs3��@mp�?0��@�z�C�3                                    Bxpd��  
w          @���p�@�  ?��@���C\��p�@Dz�@'�A�ffC�=                                    Bxpd�~            @��H��
=@r�\���a�CJ=��
=@��H?#�
@��HC8R                                    Bxpd�$  �          @�G���p�@Y���/\)����C���p�@��H���z�HCQ�                                    Bxpd��  "          @���z�@E������33Cff��z�@z=q����X��C@                                     Bxpd�p  �          @�\)���H@
=q�����0C�R���H@�33�J�H��p�Cz�                                    Bxpd�  �          @�  ���@C33��Q���RC33���@�  �
=q��=qC��                                    Bxpd�  �          @��H��\)@Q������H\)C�H��\)@�(��fff��p�Cz�                                    Bxpd�b  "          @�z���\)?�  ��=q�S�HC}q��\)@�  �����
C�                                    Bxpe
  
�          @�=q��{@
=�����Ap�C����{@�G��^�R��p�C^�                                    Bxpe�  
�          @�=q����@���
�4�C������@��
�P  �˅C�
                                    Bxpe'T  
�          @�33���\@2�\���\�3Q�C�����\@�
=�>�R���
C=q                                    Bxpe5�  
Z          @����@ff��ff�K33C����@�{�o\)���CB�                                    BxpeD�  �          @������@   ���H�2(�C�����@���;����\C�R                                    BxpeSF  �          @߮���@a�>�\)@�C#����@4z�@��A���C}q                                    Bxpea�  �          @��H���H@[�>�33@7�C�
���H@,��@	��A�
=C{                                    Bxpep�  �          @�\�θR@0  �6ff����C��θR@w
=�k���{C&f                                    Bxpe8  �          @������
@��ÿ}p���{C�\���
@��?�  AT(�C&f                                    Bxpe��  �          @�����
=@�Q�k���  Ck���
=@�(�@�A���CL�                                    Bxpe��  �          @�\)��
=@ ���s�
���RCL���
=@�p������iC�                                    Bxpe�*  �          @����G�@��\�������C���G�@n{@p�A���C\)                                    Bxpe��  �          @��
���H@��H��
=�L(�C	����H@�Q�@��A��C
                                    Bxpe�v  �          @�ff��{@�
=����{CO\��{@�Q�@ffA���C=q                                    Bxpe�  �          @�������@U��C�
���C�R����@�ff�G���Q�C&f                                    Bxpe��  �          @�����R@W��<(����Cs3���R@�p��(����{C8R                                    Bxpe�h  �          @�\)��(�@"�\��z����C��(�@�  �z���Q�C�                                    Bxpf  �          @�\����@=p��p����p�C޸����@��׿Ǯ�E��C                                    Bxpf�  �          @��H��z�@P  �Y������C\��z�@�=q��{�
�RC�R                                    Bxpf Z  �          @����33@j=q��p��Q�C���33@o\)?s33@���C}q                                    Bxpf/   �          @���Å@b�\�&ff��G�C޸�Å@U?�\)A1��C\)                                    Bxpf=�  �          @�(���
=@u?�Q�AtQ�C33��
=@(�@mp�A�(�C�)                                    BxpfLL  �          @��
��\)@J=q@�A��HC(���\)?�(�@b�\A�C&��                                    BxpfZ�  �          @���?
=q?+�@��HB�{BM��?
=q�b�\@�  BxQ�C�O\                                    Bxpfi�  �          @�\>��?���@�RB��3B�33>���Fff@�(�B�=qC�k�                                    Bxpfx>  �          @�>Ǯ?�{@�p�B�k�B�>Ǯ�7�@�
=B��C��                                    Bxpf��  �          @�׿���@33@��B���B��Ὲ���@ᙚB��Cs�                                    Bxpf��  �          @�{�B�\@#�
@��B��B�33�B�\��=q@�ffB���Ct(�                                    Bxpf�0  �          @陚?}p�@���@���BM�B���?}p�?&ff@�(�B��B=q                                    Bxpf��  �          @�����@�@�(�B���C�������Q�@�{B��C`��                                    Bxpf�|  �          @�=q� ��@{@�  B~�C
z�� �׿��
@�(�B��
CWh�                                    Bxpf�"  �          @����@0  @��HBt
=C&f�����  @���B��HCOp�                                    Bxpf��  �          @��H�(�@��@�  B|�C�f�(����@�
=B�=qCU޸                                    Bxpf�n  �          @�\��H@\)@��B�W
C	J=��H��ff@�{B�  CX�{                                    Bxpf�  �          @�  �4z�@�
@�33Bx�C���4z��=q@��B|�HCU�                                    Bxpg
�  �          @�(��(�@\)@θRBv�Cu��(���33@���B��CQ�R                                    Bxpg`  �          @�{�K�?��H@��
Bn\)CO\�K����@�p�Bq�CQY�                                    Bxpg(  �          @���4z�@5@�=qB]��C� �4z�@  @�B�33CB�H                                    Bxpg6�  �          @���Q�@=p�@�Bep�C �)�Q�5@��HB���CD��                                    BxpgER  �          @�=q�l��@>{@�(�B<�C8R�l�;�=q@�p�BgC8+�                                    BxpgS�  �          @�p���  @G
=@uB�C���  ?
=q@��B1Q�C-�\                                    Bxpgb�  �          @�G���Q�@5�@�{B�C����Q켣�
@�Q�B9C433                                    BxpgqD  �          @�
=���
?���@�=qB\)C$)���
�Y��@�G�B�C<��                                    Bxpg�  �          @�(����?   @���B/G�C.� ����33@���BG�CK��                                    Bxpg��  �          @���Q�?�ff@��RBL{C���Q����@���BOffCHB�                                    Bxpg�6  �          @���  @'�@��HB"\)Cc���  ���@��B>(�C8�                                    Bxpg��  �          @�
=���@��\@�\)B�HC!H���?�33@�Q�BR�C$ٚ                                    Bxpg��  �          @���qG�@��@3�
A��B�=q�qG�@J�H@��
B>  C��                                    Bxpg�(  �          @��R�\@���@UA�33B�\)�R�\@C�
@�BS=qC	�                                    Bxpg��  �          @�{�G�@У�?���A4��B�\�G�@��R@��B��B���                                    Bxpg�t  �          @��H���
@s�
@p��A�ffC�R���
?�
=@�
=B8�C&ff                                    Bxpg�  �          @�33���\@n�R@l(�A�\)CY����\?�33@��
B7z�C&�f                                    Bxph�  �          @�z����=�\)@�{B6�C3&f����/\)@��B�CPh�                                    Bxphf  �          @������?.{@��\B=qC-33�����G�@�Q�B  CE                                    Bxph!  �          @�\��33@�Q�@#�
A�(�C#���33@p�@�ffB��C�{                                    Bxph/�  �          @�(���=q@��R@(�A�p�C:���=q@Dz�@��
B&  C
                                    Bxph>X  �          @�=q���@�33@��A�{B�����@`  @�p�B)z�C�{                                    BxphL�            @�����\@��R@QG�Aݙ�C����\@��@�G�BC�C��                                    Bxph[�  �          @�  �E@s33@e�B��C��E?��\@��BbQ�C�)                                    BxphjJ  
Z          @�\)�e@<��@�z�BJ33C�H�e�\)@��HBpC<޸                                    Bxphx�  	`          @�33�
=@�z�?
=q@�(�B؏\�
=@�\)@��BG�B���                                    Bxph��  	�          @��[�@θR?��AIB���[�@��@��B!�B��H                                    Bxph�<  �          @��1G�@߮��G��Y��B�L��1G�@�ff@j�HA�B�                                      Bxph��  T          @��H�\)@��H@@  A���Bُ\�\)@|(�@\BU=qB�Q�                                    Bxph��  �          @��
���H@�33@VffA���B�녿��H@p��@�(�Bf�
B�p�                                    Bxph�.  
Z          @�33>�p�@�z�@�\A��B�\>�p�@�G�@���BI  B�(�                                    Bxph��  
�          @�p�>\@��?�=qA   B�33>\@��
@��B*
=B�=q                                    Bxph�z  �          @�ff��G�@�@�
Ayp�Bϔ{��G�@�{@�33B;�\B�=q                                    Bxph�   
�          @�\)�B�\@��?���AQ�B���B�\@�{@��B%z�B�.                                    Bxph��  
�          @�  ���@���?�ffAY�B����@�Q�@�  B6��B�aH                                    Bxpil  
�          @�G��0  @�p�@{A���B�\)�0  @�z�@��HB8{B�Q�                                    Bxpi  "          @�  �Q�@�p�@6ffA��RB�(��Q�@xQ�@��BD=qC=q                                    Bxpi(�  
�          @�p��S33@�{@C�
A�ffB�#��S33@e�@��BJ{C��                                    Bxpi7^  T          @�������@��@UA�Cc�����@@��RB6ffC�                                    BxpiF  �          @�{���@�33@s33A�33C
�����?�\)@�G�B:ffC!�R                                    BxpiT�  T          @�z���p�@��H@���B
�C
�
��p�?�33@���BI33C&(�                                    BxpicP  �          @�G��XQ�@�G�@r�\A�=qB��
�XQ�@(��@�p�B\��C{                                    Bxpiq�  �          @�G��|(�@���@eA�z�B�z��|(�@!�@��BM�CO\                                    Bxpi��  
�          @���@��@W�A��C =q��@*�H@�{BC��Cp�                                    Bxpi�B  �          @�\�~{@��@G
=A�B�\�~{@G
=@��BAG�C�                                    Bxpi��  �          @�G��j=q@�(�@<��A�z�B�R�j=q@XQ�@���BBQ�C	:�                                    Bxpi��  �          @��H��z�@�z�@1G�A��HB��=��z�@QG�@��
B5�HC��                                    Bxpi�4  T          @��R��G�@�  ?�G�AS�C�H��G�@Dz�@���A���C�                                    Bxpi��  �          @�����
@�z�@��A�{C�����
@O\)@��RB%33C�                                    