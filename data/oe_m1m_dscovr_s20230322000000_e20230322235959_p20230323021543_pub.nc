CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230322000000_e20230322235959_p20230323021543_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-23T02:15:43.989Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-22T00:00:00.000Z   time_coverage_end         2023-03-22T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxpi؀  �          @�Q��%@�{@q�A��B���%@1�@ǮBj�C��                                    Bxpi�&  �          @��:=q@�G�@o\)AB�W
�:=q@8��@�Q�Bc{C33                                    Bxpi��  �          @�=q�E@��\@k�A�33B����E@<��@�\)B^=qCJ=                                    Bxpjr  r          @���HQ�@�G�@���B ��B����HQ�@!�@˅Bf��C{                                    Bxpj  h          @�
=�(Q�@�z�@�\)B
=B�33�(Q�@-p�@�\)Br�C33                                    Bxpj!�  
Z          @��H�#�
@��@�(�BC�\B��f�#�
?��@�(�B�z�B��H                                    Bxpj0d  "          @񙚿333@��H@�z�B.G�Bî�333?�Q�@ۅB�.B�(�                                    Bxpj?
  
�          @�����@��H@�33BQ�B�k����@��@�p�B|33C                                    BxpjM�  �          @����\@�  @��
B�B�G���\@�@��HB�ffC}q                                    Bxpj\V  �          @��
�G�@��@�33B3z�B�
=�G�?���@�{B��C&f                                    Bxpjj�  
�          @��z�@�@���B �B�Ǯ�z�?�(�@�G�B���Cp�                                    Bxpjy�  �          @��%�@��
@k�A�RB�z��%�@@  @ǮBf��C��                                    Bxpj�H  T          @�
=�#33@��@�\)BffB�{�#33?�  @�ffB�aHC��                                    Bxpj��  �          @�Q��4z�@��@���BB���4z�?�z�@�
=B��C�=                                    Bxpj��  
�          @���*�H@�z�@���B(Q�B�G��*�H?���@ۅB�=qC}q                                    Bxpj�:  �          @�p��\(�@<��@��B�\B�W
�\(���Q�@�p�B���CjJ=                                    Bxpj��  
�          @���aG�@.{@�33B��Bי��aG���p�@�RB�Cos3                                    Bxpjц  T          @�z��\)@n�R@�Bf�
B����\)��\)@�B���C=��                                    Bxpj�,  T          @������@�=q@�BKffB����?   @�=qB��\C&�{                                    Bxpj��  T          @�{���@�ff@���BPffB�����>���@�B�(�C+L�                                    Bxpj�x  
�          @���Tz�@G
=@�
=B�p�B�  �TzῊ=q@�B��Chff                                    Bxpk  �          @�Q쿂�\@3�
@�z�B��{B��ÿ��\��@�G�B��
CjW
                                    Bxpk�  
�          @�p�����@Vff@�{Bv��B��
���ÿ8Q�@�ffB�=qCP��                                    Bxpk)j  
Z          @�\)�G�@b�\@У�Bh
=B�p��G����@���B��3CA(�                                    Bxpk8  T          @���\@N�R@��Bt\)B� ��\�O\)@�B�\)CL�{                                    BxpkF�  �          @����H@@���B��{B����H����@陚B���Cl^�                                    BxpkU\  
�          @��W�@��R@���B�B�(��W�?�Q�@�=qBsffC�H                                    Bxpkd  "          @�p���p�@J�H@��Bp�RB�LͿ�p��(��@���B�p�CH޸                                    Bxpkr�  �          @��
�9��@�=q@��BC��B����9��>��@���B���C,�                                    Bxpk�N  �          @����{@�?���AAG�C	����{@Vff@|��A�Ch�                                    Bxpk��  "          @���>{@��@��\B  B�z��>{?�@׮B|C��                                    Bxpk��  "          @�p��tz�@�p�@
=A�  B�\�tz�@~�R@���B.�RC�{                                    Bxpk�@  
�          @�ff���H@��\?�\)AaC +����H@}p�@�p�B�\C33                                    Bxpk��  �          @�\�j�H@�  @���B'�C��j�H?n{@�G�Bo��C%�                                     Bxpkʌ  "          @�33����@N�R@�=qBp�B��\���þ�@ۅB��
CG��                                    Bxpk�2  �          @����@^{@ҏ\Bt�\B��f����
=q@�p�B���CO                                      Bxpk��  T          @���/\)@l(�@�BQ  B�.�/\)�#�
@�\)B�33C4&f                                    Bxpk�~  "          @�\)�h��@�(�@���B\)B�ff�h��?�\)@�(�Bl
=C                                      Bxpl$  
�          @�
=�vff@�Q�@�  B��B�{�vff@   @�  B]
=C}q                                    Bxpl�  
�          @��H��z�@�z�@Mp�A��B�����z�@E@��RB:�Cٚ                                    Bxpl"p  
(          @������@�
=@*=qA�C�����@\��@���B(
=C�                                    Bxpl1  "          @�33���
@�p�?���AV{C����
@w
=@�G�B�C                                      Bxpl?�  �          @�  ��p�@��?B�\@��C�=��p�@��
@p��A�33C��                                    BxplNb  �          @����\)@�ff?�ff@��\B�����\)@�
=@��B�C��                                    Bxpl]  �          @������@�
=?�(�A-�B�
=����@���@���BQ�C�                                    Bxplk�  "          @��H��  @��H?��@�(�B�Ǯ��  @��\@vffA�Q�C�
                                    BxplzT  "          @�����H@���?��RA=qC �����H@��R@�
=B�RC	T{                                    Bxpl��  �          @�G���z�@��?��
A�HC h���z�@��@�p�BQ�C	#�                                    Bxpl��  
�          @�ff�^�R@�G�@eA�{B��^�R@B�\@�33BS�C
��                                    Bxpl�F  �          @����G�@�z�@{A�=qC �
��G�@hQ�@�33B\)CY�                                    Bxpl��  �          @��H���\@�\)@�RA�z�C����\@mp�@��Bz�C��                                    BxplÒ  
�          @�p���
=@�\)@�RA��HC����
=@l��@���BG�CT{                                    Bxpl�8  �          @��H���
@�(�@(�A��RC +����
@vff@�ffB��C.                                    Bxpl��  �          @��H�o\)@�\)@0��A��\B���o\)@vff@��
B9=qC�                                    Bxpl�  �          @���Q�@�\)@@  A�p�B��Q�@|(�@�ffBD�C�R                                    Bxpl�*  �          @�(��K�@θR@K�A��HB�u��K�@u�@��HBJ��C�3                                    Bxpm�  �          @��
�J�H@���@?\)A���B����J�H@�  @�
=BE��C n                                    Bxpmv  �          @���N�R@�
=@B�\A��\B�
=�N�R@z�H@�\)BF��CxR                                    Bxpm*  T          @��
�
=@���@��B�RBڮ�
=@@��@ӅBs�\B��                                    Bxpm8�  
�          @��G�@��\@���B��B�Q��G�@*�H@ٙ�B{B�aH                                    BxpmGh  	�          @����
@�G�@s�
A��B�����
@J=q@���Bj=qB�G�                                    BxpmV  �          @�Q쿴z�@�ff@�33BffB��
��z�@p�@���B�\)B���                                    Bxpmd�  �          @��H���@�\)?�33A,  Bڮ���@�
=@��HB 
=B�G�                                    BxpmsZ  "          @���/\)@���?\A;�
B�\�/\)@��@��\B!G�B�{                                    Bxpm�   "          @�
=��G�@�=q?\(�@ӅB��H��G�@���@qG�A�C�                                    Bxpm��  �          @�{���
@�����33�*{B�����
@���?��HAn�HB��                                    Bxpm�L  �          @�ff�|��@�p�������B�B��|��@�p�?�=qA Q�B�B�                                    Bxpm��  
�          @�{�s�
@��R�Z�H��B�ff�s�
@���=L��>���B�{                                    Bxpm��  T          @�33�|��@����Tz���=qB�(��|��@�{=#�
>���B�
=                                    Bxpm�>  "          @�������@��H�B�\��p�C �R����@�z�>8Q�?���B�                                    Bxpm��  �          @�Q���Q�@���-p�����C
�H��Q�@����W
=C�                                    Bxpm�  �          @��
��G�@i���-p���(�C�)��G�@��þ�
=�Mp�C&f                                    Bxpm�0  �          @�Q����@{��G����C�����@��
� �����HC{                                    Bxpn�  
�          @�Q���ff@��Z=q��p�C
5���ff@�녿�R����C�R                                    Bxpn|  �          @���ff@�  �W���p�CǮ��ff@�(��0�����C
=                                    Bxpn#"  
�          @�33����@G
=��p��ffC������@�녿��H�o\)C
�f                                    Bxpn1�  T          @��
����@%���(���RC\����@�33�(Q���ffC                                    Bxpn@n  
(          @�z���\)?У���Q��9�C!�H��\)@���w�����C��                                    BxpnO  
�          @�Q���Q�>Ǯ��
=��C0!H��Q�@1G��u���CQ�                                    Bxpn]�  
�          @��H��{?Q���p��7��C*�{��{@W���  �(�C��                                    Bxpnl`  
y          @�
=�}p�@������L33C���}p�@����j�H��\C��                                    Bxpn{  
�          @���g
=@�����
�NQ�C���g
=@��R�\(����B�                                      Bxpn��  
�          @�Q��,(��k������
C8�\�,(�@N�R��=q�X�C��                                    Bxpn�R  �          @�
=�vff@���a���C=q�vff@w
=��ff���
C�                                    Bxpn��  A          @�=q���?���
=�G�C'�����@K��@����C
                                    Bxpn��  "          @�=q���
?^�R�~�R��RC*�����
@2�\�=p�����C��                                    Bxpn�D  
�          @�  �Å>.{�z���  C2k��Å?�=q��33��33C'�                                    Bxpn��  
�          @��(Q�>��
��ff��C-
�(Q�@]p�����G��B�k�                                    Bxpn�  
�          @߮�{��G���  ǮCL�=�{@p���33�uC�                                    Bxpn�6  
�          @�=q���H��G���G��
C~s3���H?�(���\)p�B���                                    Bxpn��  T          @����z������{CNc��@����H�w{C�3                                    Bxpo�  
Z          @ᙚ�aG����
��  �n\)C4O\�aG�@C�
����?��C
=                                    Bxpo(  T          @ٙ��g�>#�
��p��e�C1� �g�@B�\��G��5p�C
=                                    Bxpo*�  
�          @ٙ��%��(����p�W
CBY��%�@%��z��d=qC��                                    Bxpo9t  
�          @�=q��G����H���Q�CrǮ��G�?�(���\)B��                                    BxpoH  �          @�녿�33���˅p�Cfٚ��33?��R����\)C��                                    BxpoV�  
�          @��Ϳ\(��1��θR� C|�=�\(�?h����
=�C��                                    Bxpoef  T          @�Q�>B�\�n�R����fC�u�>B�\�L���߮¯� C���                                    Bxpot  �          @����p  =u�+��G�C3&f�p  ?�z�����
=CT{                                    Bxpo��  "          @�Q���33?�(����\�
=C"��33@p�>�?���C �=                                    Bxpo�X  
Z          @�p���\)@*=q�����
C��\)@#�
?fff@��C�
                                    Bxpo��  "          @�Q�����@�ff@  A��RC�)����@:=q@��\B�
C�                                    Bxpo��  s          @�R��
=@���?�  A   Cz���
=@P  @W�A�33CE                                    Bxpo�J  �          @�R��G�@�=q?��AeCL���G�@P  @~�RBffC:�                                    Bxpo��  "          @�\)����@���?�(�A}�C������@c33@�(�B(�C�\                                    Bxpoږ  
G          @�z��a�@�{@��A��RB��a�@j=q@��B1�HC�R                                    Bxpo�<  5          @��H�/\)@�G�@q�BB����/\)@(��@�{BeffC\                                    Bxpo��  
Z          @�G����R@�z���R���C
�����R@��>�z�@��CG�                                    Bxpp�  "          @�  ���@��H�
�H��z�C�R���@�(�>L��?��C
�                                    Bxpp.  �          @����33@�녿8Q����RC0���33@��\?ǮAPQ�C��                                    Bxpp#�  �          @�Q���Q�@xQ�>�@\)C��Q�@G
=@Q�A�C��                                    Bxpp2z  
�          @�\��ff@���>�ff@r�\C����ff@^{@"�\A�33CE                                    BxppA   
�          @��H��  @l(�?�Q�A>�RCT{��  @p�@FffA��C�)                                    BxppO�  
(          @�G���\)@xQ�>Ǯ@P��C�H��\)@I��@�\A��RC(�                                    Bxpp^l  �          @�����{@H��?@  @�ffC���{@�@�RA�
=C\)                                    Bxppm  �          @�=q��p�@3�
?@  @�(�C^���p�@33@�
A�(�C"J=                                    Bxpp{�  
;          @�33��=q@���>���@L��CE��=q@R�\@�A�{C��                                    Bxpp�^  T          @�z����
@c�
���}p�CǮ���
@H��?�
=A[�
C��                                    Bxpp�  T          @��H��\)@*�H�}p�� Q�C�{��\)@3�
>��H@|(�C��                                    Bxpp��  �          @����=q@33�O\)��p�C ����=q@��>�ff@k�C�                                    Bxpp�P  �          @��
���@>{��z��
=C(����@.{?�p�A ��C�                                    Bxpp��  T          @��
��
=@�R�Ǯ�MC ����
=@-p��u��
=C@                                     BxppӜ  �          @�(��˅?�ff� ������C(xR�˅@Q쿋���
C!�                                     Bxpp�B  "          @�������<��
������C3������@�e���\C:�                                    Bxpp��  �          @�����33>�
=�Tz���(�C0���33@G��*�H��ffC!��                                    Bxpp��  
�          @������\�������R�6��CG����\?}p����
�>p�C'&f                                    Bxpq4  "          @�33��p���=q�n{�(�CC���p�>��H��Q����C/
                                    Bxpq�  
Z          @�Q���(����
���R�=qCD����(�?c�
��(��&��C*�                                    Bxpq+�  "          @�G�������\)��
=��RCGh�����>����H�#33C.�
                                    Bxpq:&  �          @�Q������G��mp��z�CKJ=���׾����H�  C5aH                                    BxpqH�  
�          @��H��녿�p����\�-��CG�\���?fff�����7�
C)s3                                    BxpqWr  �          @�(����ÿ�����R�&�RCF�����?k������/�C)�3                                    Bxpqf  �          @��
��ff�����
�{CGff��ff>\�����C0\                                    Bxpqt�  �          @��H����=q�g
=��{CKk���녾��R��=q�\)C7+�                                    Bxpq�d  �          @������\�(Q���{�&��CQ�)���\=#�
��(��Fp�C3�=                                    Bxpq�
  �          @�
=���Ϳ��H���H�?�RCK������?n{����M
=C(                                    Bxpq��  
�          @������Ϳ�33�%���CD�����;�p��L(��ӅC7Y�                                    Bxpq�V  �          @����=q������
��CAB���=q?�G���ff��RC)��                                    Bxpq��  �          @ᙚ����z����H�#z�C@�f��?�  ��=q�"z�C&k�                                    Bxpq̢  �          @�(����׿��\�����(�C@������?�  ��33�{C)�                                     Bxpq�H  "          @�=q��p����\��\)��C>0���p�?��������\C&�f                                    Bxpq��  �          @�(����׿0���������C:�{����?�\)�����(�C$O\                                    Bxpq��  T          @��H����:�H��z�� �C;5����?�\���
�G�C"޸                                    Bxpr:  "          @����
=�Y����G��C<���
=?��R���
�z�C&�                                    Bxpr�  �          @���ƸR����Q��	�RC5=q�ƸR?�(��q�����C"h�                                    Bxpr$�  T          @�(���p��������   C8L���p�?�ff�p  ��ffC&s3                                    Bxpr3,  �          @��
�ƸR�����Q�� �RC@�H�ƸR?=p�����HC-+�                                    BxprA�  �          @陚��  �|���_\)�噚CZE��  �޸R��
=�,CG.                                    BxprPx  �          @�{���\��=q�n{��=qC]�H���\�޸R��\)�;��CH�                                    Bxpr_  �          @�
=����AG������(�RCV�����������p��Q�C7��                                    Bxprm�  
�          @�(������$z�����Q�CO�R���ͽ�������8��C5&f                                    Bxpr|j  �          @�ff��\)�9���}p��(�CQ���\)�(����
�+33C:�)                                    Bxpr�  T          @������7��tz��G�CPff��녿&ff��\)�%��C;                                    Bxpr��  �          @�z���(��,(���  �[�
CJ!H��(���G��5��G�C@�\                                    Bxpr�\  �          @�=q�����G����3�
CF����ÿ�ff���{C>�
                                    Bxpr�  �          @����H�=q��=q�\)CH
=���H��=q�
=��  CA�                                     BxprŨ  
�          @�33��Q�����Q��<(�CH
=��Q쿰���=q��\)C?�R                                    Bxpr�N  �          @�33��\)��R=�Q�?@  CFO\��\)���R��  ��CD�                                     Bxpr��  "          @ᙚ��ff�z�.{����CE.��ff��p�����=qCB��                                    Bxpr�  �          @�=q��{�ٙ����H�\)CBG���{�h�ÿ���yp�C;�R                                    Bxps @  T          @�ff��z�Ǯ��ff�.��C8�)��z�?������Q�C�
                                    Bxps�  �          @�33���H=u�����J{C333���H@%���H�(��C+�                                    Bxps�  �          @�G���\)����z=q��
C?����\)?c�
�~�R�p�C*��                                    Bxps,2  T          @�\)��p��{�����Q�CG)��p��G��Mp���\)C:��                                    Bxps:�  T          @�\)��\)�\)�\)��ffCG
��\)�fff�B�\��33C;޸                                    BxpsI~  �          @����
=�\(���33�S�
C;\��
=<���{�o�C3�R                                    BxpsX$  	�          @�Q����H>���\)���C2�����H?��H����t(�C)��                                    Bxpsf�  
�          @�{��(�>�=q�QG��ظRC1�
��(�?�ff�/\)���C$B�                                    Bxpsup  T          @��Ǯ�
=�S�
��z�C9aH�Ǯ?���L(��ծC*T{                                    Bxps�  �          @�z���=q��p��u��
=CBB���=q?���=q�ffC.�{                                    Bxps��  T          @�{���H�\�~�R�\)C7�
���H?Ǯ�j�H�=qC$^�                                    Bxps�b  "          @�p���p�?z��u�C.(���p�@�Fff��Q�C�)                                    Bxps�  �          @������H?��������C-�����H@#33�`������C}q                                    Bxps��  T          @�ff���H���
��z��(�CA�����H?W
=��Q���HC+�                                    Bxps�T  �          @���=q�W
=���\�ffC<�R��=q?�Q��\)���C'Y�                                    Bxps��  
�          @�  ��
=�&ff��=q�/ffC;k���
=?��H�����#ffC ��                                    Bxps�  "          @�����{�p�������C=����{?��H�����C'p�                                    Bxps�F  �          @ᙚ��ff�����H�(�C9����ff?�Q��w���C%޸                                    Bxpt�  "          @߮�����p������=qC7�=���?�=q�p  �(�C$Y�                                    Bxpt�  T          @���G���R�����p�CN.��G���z���
=�,(�C7@                                     Bxpt%8  "          @�������#�
��ff�/�
CS�3���ͽ#�
����P�C4�
                                    Bxpt3�  �          @�(��������R��=q�*�CBs3����?������,��C'�{                                    BxptB�  "          @�p������G���G����C5+����?��u�
=C u�                                    BxptQ*  �          @�z����R�������H��HC?�{���R?������\��\C(�                                    Bxpt_�  "          @�������33������HCE����?����G��$�C-�                                    Bxptnv  T          @�p���=q��������!�CCB���=q?aG�����'p�C*#�                                    Bxpt}  
�          @޸R��Q��\���  CF����Q�>Ǯ�����"�C/�q                                    Bxpt��  T          @�
=���
���p  ���CKz����
���R����Q�C7G�                                    Bxpt�h  T          @�{������R�b�\����CI�R���׾��
��p����C7L�                                    Bxpt�  "          @�\)��
=�	����(��  CKp���
=>\)��(��2  C2aH                                    Bxpt��  �          @߮��  ���~{�
�HCK� ��  �.{���\�$33C5޸                                    Bxpt�Z  �          @�  ����#�
�p���
=CM�3��녿�\��Q�� �C9��                                    Bxpt�   "          @�  ��\)�.�R��G����CP�3��\)�
=q���H�0
=C:+�                                    Bxpt�  T          @������.{�s33�
=COn����#�
��(��$�C;                                    Bxpt�L  T          @�Q���p��J�H�u��
CT�\��p����
���
�0G�C?ٚ                                    Bxpu �  
�          @�Q�����`���n{�
=CX�\��������p��3CDW
                                    Bxpu�  T          @�Q����R�AG��a���p�CR
���R��������!�HC?J=                                    Bxpu>  �          @����ff�6ff�n�R� �
CP�q��ff�L����(��%ffC<Ǯ                                    Bxpu,�  
�          @�G�������vff��RCLp�����\��G�� C8�                                    Bxpu;�  T          @�=q����0���QG���z�CN8R����s33��p���C=�
                                    BxpuJ0  �          @ᙚ��Q��(���Fff�У�CL����Q�h���~{��C=�                                    BxpuX�  T          @ᙚ��(��Q��/\)��G�CG!H��(��!G��Z=q���
C9�
                                    Bxpug|  
Z          @�G���33�+������\CK����33��
=�E���33CA&f                                    Bxpuv"  T          @�\�У׿��� ����(�CD
=�У׿=p��)����C:�                                     Bxpu��  
�          @�=q��G�����
=q��Q�CH�)��G�����AG���(�C>G�                                    Bxpu�n  �          @�\���\�@���a���CQn���\�����Q����C?@                                     Bxpu�  "          @�33��
=�>�R�Y����\)CP����
=������z���
C?T{                                    Bxpu��  �          @�=q�����XQ��N{�مCT�����ÿ�������HCD@                                     Bxpu�`  �          @��H��p��u�R�\��ffCY���p���z����&{CIB�                                    Bxpu�  "          @ᙚ������H�1����\C[Y�����Q����H��CMxR                                    Bxpuܬ  "          @���������
�
=q���C]������:=q�z=q�  CR�                                    Bxpu�R  "          @߮����ff��
��33C[����3�
�o\)��CP�                                     Bxpu��  
�          @�  ��{��(���ff�mC_.��{�S�
�l����33CU޸                                    Bxpv�  
:          @߮��p����R��  �EC\����p��S33�W
=��p�CT�{                                    BxpvD  "          @�ff��p������\�J=qCX��p��;��L(��ڸRCP\)                                    Bxpv%�  T          @�����\)�z�H���H�C�CW����\)�5��Dz��ӅCOW
                                    Bxpv4�  �          @�p���
=���Ϳ�z��
=C_\��
=�g��HQ����
CX{                                    BxpvC6  
�          @ٙ���z���33�n{���\CeQ���z���z��HQ����C_W
                                    BxpvQ�  T          @�G����
���R�\)��(�Cc
=���
����
=��
=C_aH                                    Bxpv`�  �          @�����\���R�����Cf:����\��33������Cb�                                    Bxpvo(  T          @������\���H?(��@���Ce�����\��33��Q��iCd:�                                    Bxpv}�  T          @�ff��z���  ��
=�G
=C]
=��z��J=q�K���p�CT��                                    Bxpv�t  �          @�{��33�~�R���H�l��CY���33�2�\�S�
���CP��                                    Bxpv�  
�          @�(�������G���p��O�C]�=�����K��O\)��G�CU}q                                    Bxpv��  �          @�(���G���33��G��yC\�
��G��8Q��Z=q����CS�                                    Bxpv�f  T          @Ӆ��  �y���Q����C[^���  ���w���CO�                                    Bxpv�  
�          @�p����
�:=q�l����CU�3���
�u����7z�C@k�                                    Bxpvղ  �          @˅��
=�5��`  �
=CTT{��
=�z�H��z��/ffC@T{                                    Bxpv�X  
(          @�{����H���H����\CV����녿����R�%�\CEO\                                    Bxpv��  
�          @�������������QG�Cc������\���R�\��ffC[��                                    Bxpw�  4          @ҏ\���H��z�������CdaH���H���
�-p��£�C_u�                                    BxpwJ  "          @���
=�^�R�����CXh���
=�z��J=q��  CN33                                    Bxpw�  
�          @�{�����+����
����CM��������\)�1��ͮCDL�                                    Bxpw-�  
�          @��
��z��B�\�xQ���\CQk���z��33�p���\)CK)                                    Bxpw<<  
l          @ʏ\�����y����=q��RC[.�����A��+���
=CTQ�                                    BxpwJ�  
�          @�=q�����N{����u�CT�)�����
�H�8����z�CKQ�                                    BxpwY�  �          @������~{��\)�t��C^����7
=�L������CT�                                    Bxpwh.  
l          @�
=�����9����\)�ip�CO�q���׿�33�.�R�ȣ�CF�R                                    Bxpwv�  T          @Ϯ��{�A������CR=q��{��(��W����CFaH                                    Bxpw�z  
�          @˅���\�P��=#�
>ǮCT�3���\�<�Ϳ���T��CR&f                                    Bxpw�   
�          @�ff��G��:�H����AG�CP���G���
�(���  CIT{                                    Bxpw��  
�          @�G��W
=�#�
����eG�C6�R�W
=@�������H��CxR                                    Bxpw�l  	�          @�Q��dz�aG���(��\��C7���dz�@���
=�C�HCL�                                    Bxpw�  T          @�33��
=��
=�]p��ffCB{��
=>�G��hQ����C.�q                                    Bxpwθ  T          @�
=��Q쿮{�\�e�CAG���Q�녿��H��\)C9�)                                    Bxpw�^  
Z          @�=q��(�������
��ffC>���(��W
=����p�C6{                                    Bxpw�  T          @�����(���
=��{�H  C7�3��(�>W
=��z��O\)C2�                                    Bxpw��  �          @Ϯ��z�
=��{�B�\C9O\��z�Ǯ�\)����C7xR                                    Bxpx	P  "          @�Q���  ��p����H��C?�
��  ��=q�33��\)C6��                                    Bxpx�  �          @��H��G����R��\)�C�
C6����G�>��R��\)�C�C1+�                                    Bxpx&�  
�          @ҏ\�������?�  AT��CE@ �����?
=q@���CI#�                                    Bxpx5B  
�          @�ff���Ϳ�Q�?W
=@�Q�CC\)���Ϳ��<�>���CE�                                    BxpxC�  "          @˅���
�k�?��AD  C<����
��33?Q�@�{C@�                                    BxpxR�  �          @�p���<��
@*=qAȏ\C3�=����\)@=qA���C?&f                                    Bxpxa4  T          @��
����?n{@���B7�\C'�����ÿ��@��B5�RCBL�                                    Bxpxo�  
�          @�33���?�z�@:�HA���C'�����L��@H��A�RC6&f                                    Bxpx~�  
Z          @�p�����@��@Q�A�
=C�)����?�G�@<��A�
=C'+�                                    Bxpx�&  T          @����\@��
@p�A�  Cn���\@2�\@p  B=qC:�                                    Bxpx��  �          @�ff����@tz�?��HA�Q�C
h�����@'�@X��B�HC�H                                    Bxpx�r  T          @�Q���
=�e������\CYB���
=�?\)�z�����CTc�                                    Bxpx�  �          @��
��(��녾��
�@  CJ{��(���
=��G��>�RCF�3                                    BxpxǾ  T          @�����Q��8�þ.{��z�CP� ��Q��"�\�����Tz�CMǮ                                    Bxpx�d  �          @��������  �
=q��{CJ&f���ÿ���Q��^�HCF!H                                    Bxpx�
  "          @�  ���R�5�?�R@�  CP� ���R�5��!G����CP}q                                    Bxpx�  �          @����\)��z�>�@��RCG:���\)��Q쾽p��j�HCGs3                                    BxpyV  �          @�Q���p��޸R=L��?��CE��p����Ϳ.{���
CDk�                                    Bxpy�  T          @����33@.{�"�\�ҏ\C����33@b�\��z��;
=C�f                                    Bxpy�  
�          @�  ��=q@b�\�$z����C
�R��=q@�G��Y����\CxR                                    Bxpy.H  
�          @�\)��33@E�����C���33@p�׿G���C	z�                                    Bxpy<�  T          @��\�(�@��Ϳ\��B���(�@�z�>�@�33B晚                                    BxpyK�  �          @����qG�@C�
�0  ��ffC��qG�@{���(��J�\C��                                    BxpyZ:  
�          @����dz�@"�\�>{��C�=�dz�@c33������\C#�                                    Bxpyh�  	�          @��H�W
=@�{���
C��W
=@;�������33C
�f                                    Bxpyw�  "          @|(��8��@녿�G����C�f�8��@&ff�L���<��C	��                                    Bxpy�,  T          @��
�{@{������
B�  �{@��>�ff@���B���                                    Bxpy��  �          @����33@�\)���
�Y��Bݨ���33@�p�?��
A��B�8R                                    Bxpy�x  �          @�G��n�R@��ÿ�p��6�HB��H�n�R@��?\(�A Q�B���                                    Bxpy�  "          @ȣ��(Q�@��H�����=qB�p��(Q�@��H?�G�A��
B�z�                                    Bxpy��  "          @�G���=q@��ÿ�Q��V{B�\��=q@��?���AffB֙�                                    Bxpy�j  �          @�  ���@�p������K33B�����@�
=?�{A&�RB؀                                     Bxpy�  
�          @����@���P���B����@�
=�}p���\B֨�                                    Bxpy�  
�          @���\)@&ff?�R@�Q�C�{��\)@�?У�A���C��                                    Bxpy�\  
�          @�=q����@�\)?
=@�C�H����@g�@G�A�p�CO\                                    Bxpz
  "          @ƸR��=q@o\)�u�
=C33��=q@\(�?�(�A_�C��                                    Bxpz�  
�          @������@��?��@�{C����@dz�@��A��C�                                    Bxpz'N  
�          @�{��(�@�?�{Aw�C����(�?�=q@
=A�=qC&
                                    Bxpz5�  "          @����\��{���
�W
=CCc����\���H�+�����CB�                                    BxpzD�  �          @�ff���@�Ϳ�R��
=C=q���@��>���@\(�C�f                                    BxpzS@  T          @������?�ff��G����C%)���?��R>�(�@��C%�
                                    Bxpza�  T          @����z�?n{?��\A@Q�C+��z�>�{?��AiC0�q                                    Bxpzp�  "          @���p���?���A+�C5@ ��p����?n{A��C9�                                    Bxpz2  T          @ƸR���R@6ff>8Q�?ٙ�Ck����R@"�\?���AEC�                                    Bxpz��  �          @���@,��?���AAp�CB���?��R@(�A�G�C
=                                    Bxpz�~  T          @�\)��Q�?���?���A*ffC(����Q�?
=?�(�Ac\)C.#�                                    Bxpz�$            @����R��\)?��RAj�RC4�����R�(��?��ARffC:�)                                    Bxpz��  4          @�������>�
=?�33A2�HC/���������
?�(�A>{C4�=                                    Bxpz�p  "          @�����p�?��>�{@P��C*��p�?Q�?:�H@�\)C,&f                                    Bxpz�  T          @������@*�H?�z�AU�C33����?��@ffA�ffC c�                                    Bxpz�  �          @����Q�@(Q켣�
�\)CxR��Q�@�H?��Az�CL�                                    Bxpz�b  "          @������
?���>�
=@�z�C&�����
?��?h��Az�C)��                                    Bxp{  T          @Å��Q�@{?(�@���C����Q�@G�?��Ak�C�H                                    Bxp{�  �          @�G���  @=q>���@K�C^���  @z�?�G�AA�CQ�                                    Bxp{ T  �          @������H@�W
=��C� ���H@   ?#�
@ÅC J=                                    Bxp{.�  T          @�33��G�@�Ϳ�p��;33CO\��G�@ �׾�����C��                                    Bxp{=�  �          @��H���@.{�Ǯ�x��C\���@G
=��Q��c33C�H                                    Bxp{LF  �          @Å��
=@l�Ϳ
=q��z�C����
=@g�?xQ�A�C�\                                    Bxp{Z�  �          @����z=q@��R�(���ʏ\C5��z=q@��?�33A2�\C޸                                    Bxp{i�  �          @�z����
@���=q�S�C����
@,�;�z��4z�C&f                                    Bxp{x8  �          @�{���H@�{����CY����H@AG���\)�+�
CQ�                                    Bxp{��  �          @�{����?����)���˅C%������@�
��33��33Cٚ                                    Bxp{��  
�          @�ff��?��R���(�C'����?��H��Q��W33C �R                                    Bxp{�*  �          @ƸR���?�G���\��
=C*:����?�\)���R�8z�C$��                                    Bxp{��  
�          @�  ��=q?�G��!G����C(E��=q?�zὸQ�Q�C&�3                                    Bxp{�v  �          @�\)���?^�R>�?���C+�����?@  >�@�\)C-\                                    Bxp{�  "          @�Q����H?�
=?�R@�p�C)����H?Y��?��
A�C,
                                    Bxp{��  �          @�  ����?�\)?O\)@�  C'=q����?p��?��
A=p�C+�                                    Bxp{�h  T          @�z����H?�  >���@C�
C#J=���H?��R?z�HA�
C%��                                    Bxp{�  
�          @�����>����ͿxQ�C/c����>�=�Q�?c�
C/aH                                    Bxp|
�  �          @��H��?�\)�����
C ����@{��ff�
=C�=                                    Bxp|Z  T          @���  ?��R��Q��Z�\C%� ��  ?�
=�B�\���
C!s3                                    Bxp|(   �          @����@ ���1�����CY����@;���ff���RC�3                                    Bxp|6�  �          @�Q�����?�{��\)��Q�C&�q����?�p����H�3
=C!�                                    Bxp|EL  �          @��
��\)?�R��(����RC-�=��\)?�=q��=q�p(�C&�                                    Bxp|S�  T          @�����(�@.{�(���
=C
=��(�@Vff�z�H���C�{                                    Bxp|b�  �          @�Q����@ ����R���Ch����@J�H�����"=qC��                                    Bxp|q>  �          @�p�����@dz��z�����C������@��þ����qG�C
B�                                    Bxp|�  �          @�p���=q@H�ÿ�33����C�f��=q@hQ�\)���HC                                      Bxp|��  T          @�p���
=@.�R��ff���Ch���
=@Mp��!G����\Cn                                    Bxp|�0  �          @�\)���?�=q�333�ظRC(�����@�
�
�H��  C�                                    Bxp|��  
(          @�G���z�?Y���C�
��=qC+  ��z�?���� ������C )                                    Bxp|�|  T          @�����33?�=q�XQ����C%W
��33@ ���'���  C                                    Bxp|�"  
�          @���Q�?��
��=q�%\)C'+���Q�@\)�XQ��
=C)                                    Bxp|��  	�          @�{�G
=?�=q����\�C(��G
=@U�����%��C�R                                    Bxp|�n  "          @ƸR�w�?�����
�A��C!\)�w�@<���q��z�C��                                    Bxp|�  �          @�ff��{?�����\�'=qC&�
��{@!G��XQ��=qCu�                                    Bxp}�  
�          @�p����H?B�\�k��C+���H@�G���p�C�H                                    Bxp}`  �          @�p�����?��w
=�(�C-�H����?�Q��W��
=C޸                                    Bxp}!  
�          @Å��  >�Q���33�(��C/n��  ?�\)�j=q��RC}q                                    Bxp}/�  T          @�33���H?�=q���\�6�\C%8R���H@'��g
=��Cff                                    Bxp}>R  �          @Å�p  ?�
=��\)�=��C޸�p  @N�R�a����CG�                                    Bxp}L�  
�          @\�|��?�����=q�5�
CE�|��@Dz��Z�H�	��C(�                                    Bxp}[�  �          @�(����=�G��(Q���
=C2�H���?�=q�����(�C(��                                    Bxp}jD  "          @��
���R?s33�����C*)���R?����z�����C!��                                    Bxp}x�  "          @�{���?�=q�7
=���C#+����@!������C33                                    Bxp}��  
�          @�33��\)>k�=L��>�C1Ǯ��\)>L��>�?��RC2)                                    Bxp}�6  T          @�G����
>�(�?�p�A=�C/ٚ���
�u?�ffAH(�C4��                                    Bxp}��  
(          @����\)�˅@   A�Q�CD.��\)�p�?��\AB=qCI�q                                    Bxp}��  "          @�G���Q��G�@!�A�(�CC�q��Q���?��A��\CK��                                    Bxp}�(  �          @��Ϳ�z�?G������!��C���z�?�p��}p�����C
�f                                    Bxp}��  "          @�  �@`�����\�:�RB��@�33�0  �ۮB��                                    Bxp}�t  "          @�\)�>�R@G
=�{��)z�C�{�>�R@��\�!G�����B�#�                                    Bxp}�  
�          @��,��@\)�Y����RB�\�,��@��R�ٙ���z�B�{                                    Bxp}��  �          @���<��@s33�Z=q�(�B��q�<��@��ÿ����G�B�p�                                    Bxp~f  �          @�=q�)��@W��z=q�)p�C :��)��@����H����B�ff                                    Bxp~  
�          @�33��
@��
�C33���
B���
@���G��Hz�B�{                                    Bxp~(�  
�          @�33����@�p��j�H�$��B�����@�
=��z����\B�                                    Bxp~7X  T          @�33���@tz����H�3{B��)���@�G������
B��                                    Bxp~E�  
�          @���>W
=@U���R�_�
B�>W
=@���Z�H�G�B�\                                    Bxp~T�  T          @�ff�u?���p�.B�  �u@j�H�����D�HB�=q                                    Bxp~cJ  �          @��R�	��@(����(��\ffC&f�	��@�
=�fff�{B���                                    Bxp~q�  	�          @Å��  @!G���  �nB����  @����  �$G�B���                                    Bxp~��  �          @\�!�@,(������Q=qC8R�!�@�
=�_\)��B���                                    Bxp~�<  �          @�  ���@S33��(��<=qB�����@�z��:=q���HBꞸ                                    Bxp~��  �          @�����H@���>�R���HB�{��H@�z῝p��B�HB�u�                                    Bxp~��  
�          @�z�����?��?�z�A��C%n����?!G�@G�A�C,�H                                    Bxp~�.  "          @��R���Ϳ�{@aG�B��CF33�����!�@3�
A�(�CS^�                                    Bxp~��  T          @�  �l��=���@�\)BC{C2h��l�Ϳ�z�@\)B4��CH�=                                    Bxp~�z  �          @���.{>�ff@��HBvffC*�
�.{���@�p�Bh�
CO                                    Bxp~�   T          @�ff�33?�{@�{B�z�C���33�aG�@�\)B��{CKL�                                    Bxp~��  �          @��Ϳ�?�z�@��
B�G�C�����z�@���B�aHCE�=                                    Bxpl  T          @�=q�^�R@\)@�\)B��qB�aH�^�R>�  @�B���C$
                                    Bxp  �          @��H�-p�?�@�ffB[
=C�q�-p�>\)@�=qBwffC1\                                    Bxp!�  "          @�33�
=@4z�@���BM\)B��\�
=?��@��
B�(�C��                                    Bxp0^  
�          @�Q��_\)?���@�BB�HC"��_\)���@�=qBKG�C:�H                                    Bxp?  "          @�\)�R�\?�p�@��HB;�C�f�R�\>�(�@���BW
=C,�)                                    BxpM�  
Z          @�
=����?\(�@��\B���Cٚ���ÿ��@�G�B��=CR=q                                    Bxp\P  
          @��R��\)>k�@�
=B�aHC(����\)�ٙ�@��RB�G�Cl��                                    Bxpj�  3          @�
=�(�?k�@���Bz
=CT{�(��aG�@���Bz��CG��                                    Bxpy�  
�          @�Q��Tz�@	��@�z�B9=qC��Tz�?z�@�(�BW�HC*
                                    Bxp�B  
�          @����\)@   @P��Bp�C����\)?@  @p��B$�\C*                                      Bxp��  �          @��R�X��@�33@�A���C���X��@I��@Tz�B��C	�                                    Bxp��  
�          @��Ϳ�@dz�@G�B#=qB���@G�@��Br��B�                                      Bxp�4  �          @�z��)��@�\)@
=qA��B�\�)��@N{@_\)B�RCaH                                    Bxp��  
�          @�z��e@{�?�33A�{Cn�e@G�@:=qA�p�C�                                    Bxpр  "          @���e@g
=@�\A�p�C�)�e@&ff@X��Bz�C�                                    Bxp�&  
�          @�  �p��@�>��R@J=qC���p��@tz�?�p�A�
=C}q                                    Bxp��  
�          @�\)�Z�H@��
>�\)@@��C���Z�H@qG�?�
=A�C.                                    Bxp�r  "          @�ff�x��@��׾����UC\�x��@x��?���A/�
C�                                    Bxp�  �          @����[�@����Q����HCu��[�@���#�
��33B��R                                    Bxp��  �          @��\��G�@dz������RC�\��G�@c33?333@�p�C�                                    Bxp�)d  "          @����@\(���=q�UC&f��@l(����
�O\)C33                                    Bxp�8
  T          @�33����@fff��
=���RC�����@�G��
=��=qC�q                                    Bxp�F�  T          @��R��\)@S33���z�C����\)@s�
�W
=��RC��                                    Bxp�UV  �          @�
=�p  @XQ��AG�����C	�3�p  @�ff��Q����
C��                                    Bxp�c�  T          @���2�\@c33�J�H�z�C )�2�\@�p������33B�                                     Bxp�r�  
�          @��\(��)��@r�\B$\)CY�\�\(��qG�@*�HA�
=Cc��                                    Bxp��H  T          @�{�Q���>�{@�=qC@���Q�!G�>�@Dz�CB�                                     Bxp���  T          @�z����@AG������H�\B������@���O\)��B��f                                    Bxp���  "          @���HQ�@ff�����BffC.�HQ�@Z=q�Tz���RC�\                                    Bxp��:  �          @���}p�@{�U���HC��}p�@\(���
��  C\                                    Bxp���  
Z          @�Q���G�@��4z���\C�3��G�@6ff��p���=qC޸                                    Bxp�ʆ  T          @�(��У�@vff������HB��H�У�@��z�H�;\)B�ff                                    Bxp��,  �          @�  � ��@[��\��G�B�W
� ��@o\)��z��k�B��                                    Bxp���  T          @�33��
=@Q녿�  �p�C.��
=@[�>��?�p�C�                                    Bxp��x  �          @����@W
=���m�CE��@h�þk���C                                      Bxp�  
(          @�ff�dz�@�(����R�tz�C���dz�@�z�#�
���C(�                                    Bxp��  T          @��
�hQ�@�=q������C� �hQ�@�ff�Ǯ�w
=C=q                                    Bxp�"j  "          @��
�h��@�(���R����C���h��@��
�(����B��                                    Bxp�1  �          @�Q�����@�  �����)�Cff����@�(�>�Q�@`��C�{                                    Bxp�?�  
�          @������@N�R��R���
C�����@r�\����$z�C
�)                                    Bxp�N\            @�G����@O\)�u��CW
���@E�?�G�A�HC��                                    Bxp�]  
�          @����@?�(�A�ffC����?�z�@�
A���C%!H                                    Bxp�k�  �          @�ff��G�@ff?
=q@�G�C {��G�?�ff?���A3
=C"�                                    Bxp�zN  3          @�  ����?n{�k����C+n����?s33=��
?B�\C+33                                    Bxp���  "          @ȣ���{>�ff�L����=qC/�
��{?.{�(���=qC-��                                    Bxp���  �          @ȣ��Å>��ÿ�  �8z�C0���Å?8Q쿈���=qC-G�                                    Bxp��@  u          @ȣ���
=��=q��ff��{C6����
=>��ÿ����G�C0�{                                    Bxp���  e          @�Q���{��
=��33�vffCF�R��{����(����\C@                                    Bxp�Ì  
(          @��H��{�	���˅�s\)CI�=��{�\�����=qCC�{                                    Bxp��2  
�          @����Q���\�������\CM����Q��z���R����CGB�                                    Bxp���  
�          @�(���G�@XQ��w
=�8�\B�.��G�@�{�#�
����Bٔ{                                    Bxp��~  
�          @�\)��@�=q�s�
�\)B�#���@�=q�p����B�aH                                    Bxp��$  �          @�Q����@�(��h����RB�R���@����\���B�.                                    Bxp��  �          @�녿�\@���S33���HBڣ׿�\@�33��  �]�Bսq                                    Bxp�p  �          @�\)��
=@�G��4z���(�B�{��
=@���z�H�  Bʀ                                     Bxp�*  �          @Ǯ���@��\��\)�9(�B�8R���@����<(���RB�z�                                    Bxp�8�  "          @ȣ����@o\)��Q��8�B�
=���@�\)�E��=qB�R                                    Bxp�Gb  "          @�Q��(�@dz���(��?z�B�{�(�@�33�P  ��G�B�\                                    Bxp�V  �          @�  �G�@\(����Bz�B���G�@�  �U��RB�.                                    Bxp�d�  
�          @�Q�� ��@hQ�����8\)B�\� ��@���<(���G�B�p�                                    Bxp�sT  
�          @�=q�7�@>�R���;�HC��7�@�\)�P���(�B�W
                                    Bxp���  �          @�Q��*=q@^�R��G��*�B���*=q@��\�.�R��ffB�p�                                    Bxp���  �          @���O\)@@  ���
�,
=C	5��O\)@�z��=p���
=C                                     Bxp��F  "          @�  �_\)@l(��B�\���Cc��_\)@��R��p����C                                     Bxp���  �          @�  ��(�@e������HC
�R��(�@������8(�C�f                                    Bxp���  �          @�{�\)@P����G��?�RB����\)@����R�\��Bힸ                                    Bxp��8  "          @��
�!�@!G���  �_�C
=�!�@�G���ff�%p�B�
=                                    Bxp���  "          @�=q��ff@/\)�����j��B�k���ff@��������*=qB�ff                                    Bxp��  T          @��/\)@=q��z��O=qC
�)�/\)@o\)�j=q�
=B�ff                                    Bxp��*  �          @�\)�j=q?�����H�F{C*���j=q?�
=�|(��.p�C8R                                    Bxp��  �          @���Fff    ��G��dQ�C4��Fff?��R����SCff                                    Bxp�v  �          @������������
�2Q�CBٚ����>����  �9�C0Q�                                    Bxp�#  �          @�z���?�z����H�j�C�{��@E����H�8��C 8R                                    Bxp�1�  
�          @�{�5?�
=����j��C���5@+���{�B��C��                                    Bxp�@h  �          @�
=�_\)?�p���=q�IQ�C�R�_\)@4z��x���#�C
=                                    Bxp�O  �          @�z��G�?�G�����Q�HC�\�G�@G
=�x���%�
C�                                    Bxp�]�  �          @���1�@ ����\)�XG�C!H�1�@W��xQ��&��C��                                    Bxp�lZ  �          @������@������\�\C�R���@g
=�s�
�%\)B�L�                                    Bxp�{   �          @���\)@.{�����P33C���\)@~�R�^{���B��H                                    Bxp���  T          @�
=��\)@Z�H��\)�G(�B�Ǯ��\)@�(��N{�
=Bڨ�                                    Bxp��L  
�          @�p���=q@hQ����R�F�\B����=q@�=q�H�����B�B�                                    Bxp���  �          @�{��ff@Fff��{�a�HB�G���ff@��R�p���z�B�ff                                    Bxp���  �          @�Q��G�@a������==qB� �G�@�ff�G����B�k�                                    Bxp��>  �          @����\)@W
=���>p�B�p��\)@����L��� �HB�                                     Bxp���  �          @�ff���@=p���ff�TG�B��3���@�=q�tz��Q�B�.                                    Bxp��  �          @�G��=q@%��{�^�
C޸�=q@�G�����%�B�=                                    Bxp��0  �          @ə���
@,(���G��eQ�B����
@�p���\)�)(�B�3                                    Bxp���  T          @ȣ׿0��@W
=��ff�bQ�B�{�0��@����|(��
=B�\)                                    Bxp�|  �          @�{��G�@@  ��  �`p�B�uÿ�G�@��
�w
=���B�(�                                    Bxp�"  �          @�ff��ff��Q��{��%�\C50���ff?���qG��C&8R                                    Bxp�*�  �          @�
=��Q�>������0ffC-����Q�?�
=�tz��ffCu�                                    Bxp�9n  T          @����
?��\�p  ��HC#�����
@�L���=qC��                                    Bxp�H  �          @�Q��\)?\�w
=�(��C)�\)@'
=�N�R��\C�\                                    Bxp�V�  T          @�����{?�\)�r�\� �C�{��{@+��HQ�� �RCff                                    Bxp�e`  T          @���r�\@%��n�R�z�C���r�\@e��2�\��G�C��                                    Bxp�t  �          @���%@�(��<(���Q�B�.�%@�����  �mG�B�8R                                    Bxp���  T          @�33�8Q�?������SC�q�8Q�@J=q�r�\�&Q�C\)                                    Bxp��R  �          @���P  ?��
����C\)C5��P  @=p��aG���RC	�H                                    Bxp���  �          @�{�h��?Tz��xQ��8=qC'��h��?����]p��C��                                    Bxp���  T          @�z����?�  ��Q�L�C�R����@
=��Q��[(�C #�                                    Bxp��D  T          @��R��Q�?�  ��Q��B�Q쿘Q�@N�R��Q��R��B�u�                                    Bxp���  �          @�\)���?Tz���33�~��C �����@33��(��Y��C
=                                    Bxp�ڐ  �          @��R���?����\�~p�C'����@���{�_{C\)                                    Bxp��6  T          @�G��  �(��Q��6z�CC(��  =��p��=�HC1                                    Bxp���  T          @�(��|���u������33C`��|���K��,�����CZ�\                                    Bxp��  �          @�p���Q��aG��(�����HC]J=��Q��K�����Q�CZY�                                    Bxp�(  �          @�
=�hQ��P  �޸R��G�C]ٚ�hQ��&ff�'����HCW��                                    Bxp�#�  �          @��R�_\)�{����0�CW:��_\)��33��ff�O�CF&f                                    Bxp�2t  �          @����dz��
=��{�5�CR�H�dz�L�����
�N�\C@��                                    Bxp�A  �          @�(���ff��{�Z=q��
CI����ff�G��s33��C=xR                                    Bxp�O�  "          @�������   �A���CP���녿�G��g����CFQ�                                    Bxp�^f  �          @����I���33�����IQ�CU
=�I���&ff���c�\C?�)                                    Bxp�m  T          @�
=�U��33��(��A{CS�{�U��0����G��Zp�C?��                                    Bxp�{�  
�          @��H�hQ��aG��X���G�C`!H�hQ������Q��1�RCUc�                                    Bxp��X  
�          @�Q��:�H����A���(�Cl���:�H�Z�H��ff�,{Ce��                                    Bxp���  �          @��
�z�H?�33�xQ��-��C#���z�H@{�XQ���C}q                                    Bxp���  
�          @����@
�H�p���Q�C����@J=q�>�R���C��                                    Bxp��J  �          @�
=����@���n�R�ffC:�����@Z=q�7���G�Ch�                                    Bxp���  �          @�G���(�@���  ��C�
��(�@H���N�R����CaH                                    Bxp�Ӗ  �          @˅��Q�@!G��~{�  Ch���Q�@c33�E���RC33                                    Bxp��<  �          @�z���  @
=�����"  C���  @\���S�
����C�                                    Bxp���  �          @�=q��
=?���������C+���
=@<���U���z�C�\                                    Bxp���  
�          @�����Q�?s33�hQ���
C)E��Q�?�Q��Mp����HC�=                                    Bxp�.  �          @����p�?�{��  ��{C$0���p�@�\��(��7
=C @                                     Bxp��  �          @ʏ\�~{<��
�j=q�*z�C3���~{?���`  �!�\C%�                                    Bxp�+z  
(          @����p��?�\)����G�\C#Y��p��@�������+CxR                                    Bxp�:   "          @�=q�y��?�Q���33�:ffC�\�y��@J=q�w��p�C�3                                    Bxp�H�  T          @Ǯ�\��?�
=�����O�C\�\��@@  ��\)�+
=C\                                    Bxp�Wl  �          @ə��aG�?�\)��Q��S�RC���aG�@.�R��p��2�C=q                                    Bxp�f  
�          @�(��W
=?:�H��z��a{C'���W
=@	����\)�G�\CY�                                    Bxp�t�  T          @Ϯ��?��H���H�6
=C���@<(��{��G�C�H                                    Bxp��^  �          @Ϯ����@   ���
�+�C�����@H���h���	�HC�)                                    Bxp��  T          @�{���\?�Q�����*�C%u����\@
=�vff�  C�R                                    Bxp���  T          @�z�����?��
�����  C'�)����@��fff�	�C!H                                    Bxp��P  T          @�����H@#�
�����&�
C�����H@j=q�Z=q���C
0�                                    Bxp���  
�          @Ϯ���@Q���(��33C�H���@\(��S�
���HC�                                    Bxp�̜  �          @�Q��{�@
=����4��C\�{�@c33�qG����C	�                                    Bxp��B  �          @�
=�g�@&ff���9��CW
�g�@s33�qG���HC��                                    Bxp���  �          @�(��e�@6ff��ff�0C� �e�@~{�^�R�G�C
=                                    Bxp���  �          @�  �l(�@:=q���R�.{C�R�l(�@����^�R��HCn                                    Bxp�4  �          @����G�@\����  ���Cz���G�@�p��8����  CaH                                    Bxp��  �          @ָR�tz�@s�
�\)�33C��tz�@�Q��2�\��Q�C �3                                    Bxp�$�  �          @�ff��G�@w��Q����HC	�R��G�@��
���z�C�)                                    Bxp�3&  �          @�Q���@�
=�:�H��ffCff��@����33�a�CW
                                    Bxp�A�  
�          @�Q���Q�@�33�<���θRCh���Q�@������`  Cz�                                    Bxp�Pr  �          @�ff��\)@���&ff��
=CB���\)@�G����\�.�\C                                    Bxp�_  �          @Ӆ��G�@��
�Mp���RCp���G�@��H�������
C�
                                    Bxp�m�  �          @Ϯ���@�=q�7���{C�����@�ff��33�k
=Cz�                                    Bxp�|d  �          @�z���=q@s33�e�(�C	  ��=q@�(��=q���RCO\                                    Bxp��
  �          @�
=��(�@e��z=q�=qC!H��(�@�Q��1��Ù�C�                                    Bxp���  �          @�p���  @��R�7�����C8R��  @��\��{�`Q�CJ=                                    Bxp��V  T          @�ff�w
=@�ff��z��D��B�.�w
=@�z�=u>��HB�=q                                    Bxp���  T          @�33�a�@��R�}p��
�\B����a�@���?�@���B�33                                    Bxp�Ţ  �          @�Q��n�R@��R��=q���B�.�n�R@��>\@UB�.                                    Bxp��H  "          @�=q����@���z�����CL�����@�(��333��z�C �                                    Bxp���  T          @�=q�j=q@�
=��R��
=B��3�j=q@�z�L����Q�B�W
                                    Bxp��  �          @�G��g�@��R�����B�G��g�@��
�G��ڏ\B�                                      Bxp� :  �          @����b�\@�p������z�B���b�\@�z�z�H�	�B��                                    Bxp��  �          @�\)�k�@�\)������
=B����k�@�G���G��xQ�B���                                    Bxp��  �          @����!G�@�p��%��p�B�q�!G�@�p����
��B�                                    Bxp�,,  �          @�
=��@��\�,������B�=q��@����z��'
=B�aH                                    Bxp�:�  T          @Ϯ����@����>�R��33B�aH����@�(�����E�B�\)                                    Bxp�Ix  �          @�
=��
=@�
=�<����=qB�(���
=@�녿�\)�E�B�                                      Bxp�X  �          @�(��{@�
=�  ��
=B�33�{@�(��333��z�B�k�                                    Bxp�f�  �          @�z��G�@��\��
����B�G��G�@����H��p�B��f                                    Bxp�uj  �          @��G�@��������B�z��G�@\�aG�����B���                                    Bxp��  �          @�ff�Tz�@��R�l���Q�B�#׿Tz�@�Q�����{B�                                    Bxp���  �          @�ff�(��@���xQ��{B£׿(��@��R�Q���G�B��{                                    Bxp��\  �          @����R@�������B�=q��R@��\�%��
=B�\                                    Bxp��  T          @�zῌ��@���e��
B������@������B�
=                                    Bxp���  �          @��H��=q@�G��k��
=B�{��=q@��H�p���33B��)                                    Bxp��N  �          @��H�W
=@���r�\�(�B�(��W
=@�=q�����RB�z�                                    Bxp���  �          @�33�z�H@����dz����Bə��z�H@�������HB��)                                    Bxp��  �          @ʏ\�&ff@�ff�xQ��ffB��&ff@���������B��
                                    Bxp��@  "          @��H�8Q�@�33��Q���B��8Q�@���%����B�W
                                    Bxp��  "          @���z�@�
=��33�7B��)��z�@�G��Tz���{B���                                    Bxp��  �          @����p�@�������B��B��)��p�@�p��dz���\B�Ǯ                                    Bxp�%2  T          @����@����R�=Q�B�LͿ�@����\(��
=B�
=                                    Bxp�3�  
�          @Ϯ�Q�@�  ���\�'�HBǅ�Q�@�
=�;���G�B�B�                                    Bxp�B~  �          @�\)�=p�@�  ��33�5  Bƙ��=p�@����P����p�B��                                    Bxp�Q$  �          @�z�z�H@�  ���/�Ḅ׿z�H@�  �E��B�B�                                    Bxp�_�  
�          @�Q�u@������.  B̊=�u@����?\)��RB�8R                                    Bxp�np  �          @����@��\��Q��/  B����@�G��>�R��\)B�=q                                    Bxp�}  �          @ə��:�H@�����G��9
=B�z�:�H@�=q�P����=qBøR                                    Bxp���  
�          @�p��&ff@����\)�;(�BŸR�&ff@��O\)��ffB�=q                                    Bxp��b  �          @�{��Q�@e��\)�;�\B�z��Q�@�=q�I����RBܞ�                                    Bxp��  �          @�ff����@l(����4G�B��
����@����E����B�{                                    Bxp���  T          @����2�\?k���=q��{C!���2�\?u���
��{C �q                                    Bxp��T  "          @�
=�`  �)��@hQ�Bp�CY��`  �`  @4z�A�G�Ca                                      Bxp���  
�          @�
=�h����@uB)�CR=q�h���C33@J=qB��C[�                                    Bxp��  "          @�\)�fff���@g�B"�CSu��fff�Dz�@;�A�  C\n                                    Bxp��F  T          @���X����@�  B;
=CT  �X���I��@c33B33C^޸                                    Bxp� �  T          @�(��c�
����@�ffBG33CF�f�c�
��
@\)B,�RCT�q                                    Bxp��  T          @�=q�|�Ϳ}p�@�33B4��CB  �|��� ��@mp�B�\CN�q                                    Bxp�8  "          @�����  ��
=@g�B�HC9L���  ���@X��BffCD��                                    Bxp�,�  �          @����>��@g
=B�C.�������@g
=B�C9�)                                    Bxp�;�  "          @���z�?Y��@j=qB�C)���z��@p  B"
=C5�
                                    Bxp�J*  �          @��
���ͿQ�@���B.G�C?0����Ϳ�=q@k�B(�CK�=                                    Bxp�X�  �          @����h�þ�\)@�ffBJ�C8n�h�ÿ�@�\)B=33CIJ=                                    Bxp�gv  
�          @�\)�6ff?�33@�33B[�CaH�6ff?
=@�Br�\C(aH                                    Bxp�v  �          @����"�\@
=@�Q�Bb��CG��"�\?@  @�z�B\)C#��                                    Bxp���  �          @����!G�@�R@�BSC}q�!G�?�Q�@��BvffC��                                    Bxp��h  �          @�z��&ff@ ��@�(�BP�C�q�&ff?��R@��
Br�\Cz�                                    Bxp��  �          @����>{@'�@�(�B8�
C
�H�>{?�(�@��BZG�C�H                                    Bxp���  �          @�z��:�H@$z�@�  B7(�C
�
�:�H?��H@���BX�RCT{                                    Bxp��Z  �          @�ff�333@C�
@x��B,z�Cu��333?�(�@�G�BS�C�                                    Bxp��   �          @��R�b�\?���@n�RB/��C)�b�\?\)@�Q�B@��C*�3                                    Bxp�ܦ  �          @�33�z�H�(��@s�
B/  C=���z�H��{@`��B��CJJ=                                    Bxp��L  �          @�33�l(��<��@=p�B ffCZ���l(��g
=@
=A�(�C`c�                                    Bxp���  
(          @��
�H���;�@C33B��C_
=�H���g
=@��A�{Ce                                      Bxp��  
Z          @����mp�����@Z=qB"ffCIL��mp����@;�B
=CSk�                                    Bxp�>  T          @��
�h��>.{@H��B#(�C1J=�h�ÿ!G�@EBz�C=�
                                    Bxp�%�  �          @�G���@N{@z�A�B񞸿�@%@4z�B$��B�#�                                    Bxp�4�  �          @��Ϳ�{�B�\@��B��\C9�R��{���@�p�Bv�CV��                                    Bxp�C0  
�          @�  �L��@AG�?�A�ffC��L��@�@'�BQ�C�                                     Bxp�Q�  T          @����=p�@�  ?�\@��B����=p�@o\)?�G�A�=qC W
                                    Bxp�`|  "          @�(���  @�  �	����p�B�  ��  @�p������N{B�p�                                    Bxp�o"  
Z          @����@mp��QG��=qB�p����@����R����Bԣ�                                    Bxp�}�  
�          @��Ϳ���@{��Fff�=qB�q����@��\� ����{B�                                    Bxp��n  �          @��H����@dz��[�� �B�8R����@�=q��H��=qB�=                                    Bxp��  T          @�Q��p�@a��o\)�*�B�\��p�@���.�R��B��
                                    Bxp���  �          @�p��\@vff�u�+Q�B��\@�{�0  ��RB��
                                    Bxp��`  �          @��Ϳ��@l(��s�
�+�B�����@����0����z�B�(�                                    Bxp��  
�          @��\�Ǯ@Mp������Hp�B���Ǯ@��Tz����B��                                    Bxp�լ  T          @��ͿL��@333����i��Bӽq�L��@y���xQ��1{B�{                                    Bxp��R  �          @��;�{@p�����ffB��쾮{@i�������E��B��3                                    Bxp���  "          @����W
=@w��vff�0B̊=�W
=@��R�0����B�B�                                    Bxp��  �          @��\�p��@mp���Q��9p�BЅ�p��@��H�<��� ��B�.                                    Bxp�D  �          @�33��\)@B�\��Q��XQ�B�  ��\)@�(��u�#z�B޸R                                    Bxp��  
�          @����?����(��iz�C5��@B�\��  �?��B�.                                    Bxp�-�  �          @����\?�Q���  �z��Cu���\@����G��W\)C��                                    Bxp�<6  �          @�����=q?!G���(��3C ����=q?������u=qC�R                                    Bxp�J�  �          @�p�� �׿W
=���\��CJ��� ��>�ff��(�L�C'Q�                                    Bxp�Y�  T          @�G�����?(�����C:����?�������|{C ��                                    Bxp�h(  T          @�{��
=?�����Q�C�\��
=@$z����]{B�k�                                    Bxp�v�  �          @�=q�ff@L(��u�4z�B��3�ff@����:�H��\B��)                                    Bxp��t  
�          @�\)�{@e�U���B��{@�������p�B��                                    Bxp��  �          @��R��=q@~�R�[���HB��쿪=q@��R���˅B�p�                                    Bxp���  T          @��Ϳ�ff?�{���R��B��
��ff@>{��33�NB�(�                                    Bxp��f  T          @�G��\)?xQ���z�u�B��H�\)@{���� B�G�                                    Bxp��  T          @���I��@I���\���
=C�3�I��@z=q�#�
��{C �{                                    Bxp�β  
�          @���Q�@:=q�X����HC
}q�Q�@j=q�#�
��ffC�                                    Bxp��X  �          @�=q�X��@:=q�Tz��p�Cff�X��@h���\)����C��                                    Bxp���  �          @����N�R@1G��Z=q���Cs3�N�R@a��'���p�C�                                    Bxp���  "          @����_\)@U��,����Ck��_\)@x�ÿ�����C�                                    Bxp�	J  �          @���l(�@l�Ϳ���]�C���l(�@y�������`��Cs3                                    Bxp��  T          @�Q��z�H@U��}p��-C�3�z�H@^{����Q�C
��                                    Bxp�&�  T          @���g�@i���z�H�,��CǮ�g�@q녽L�Ϳ   CǮ                                    Bxp�5<  
Z          @�
=�p  @\�Ϳ�����
C	\)�p  @^�R>��
@g�C	!H                                    Bxp�C�  
�          @�ff����?
=q@��B�B�C#xR���ÿ&ff@�
=B�u�CG��                                    Bxp�R�  �          @��R��{?c�
@�B�� CǮ��{��Q�@�  B��{CB��                                    Bxp�a.  
�          @�ff�   �B�\@�(�B��CH���   ����@�B�k�Cz�                                    Bxp�o�  �          @�  �\>�ff@�
=B��=C#���\�J=q@�p�B��CO�                                    Bxp�~z  �          @�Q쿌��>�@��B�� C8R���ͿJ=q@�=qB�{CW�                                    Bxp��   �          @��׿Q�?�  @��\B�aHC)�Q녾�\)@�p�B�
=CF�{                                    Bxp���  �          @��ÿ��?��R@�\)B��{B�uÿ��>aG�@�ffB�  CQ�                                    Bxp��l  �          @��R���@��@�  Bt=qB�=���?�=q@�p�B�
=C�R                                    Bxp��  T          @��
�@�H@uBB�C��?�33@�=qBf�
C\                                    Bxp�Ǹ  T          @��
�#�
@\(�@c33B7p�B�\)�#�
@p�@�G�Bpp�B��H                                    Bxp��^  �          @��Ϳ���@��=u?Q�B�{����@}p�?���Ad��B�L�                                    Bxp��  �          @�(���@U@�
A�
=B�.��@*�H@C�
B&�C5�                                    Bxp��  T          @��R��\?�{@�
=B��
B�p���\>��@���B��=C#E                                    Bxp�P  �          @�ff���?�p�@�Q�B���Bݙ���׽#�
@��B���C8�q                                    Bxp��  �          @��R��p�?�\)@�\)B�B�B�.��p�=���@�p�B��=C$�f                                    Bxp��  �          @�{<�?��@�Q�B���B�(�<�?\)@���B�=qB�#�                                    Bxp�.B  �          @�p�>�(�@\)@�\)BrQ�B�aH>�(�?��@�ffB��
B��                                    Bxp�<�  T          @��H>�33@0  @uBX�RB�u�>�33?�(�@�z�B�p�B���                                    Bxp�K�  �          @��?��?��@��B{(�Bp33?��?J=q@��\B�=qB��                                    Bxp�Z4  T          @��\?c�
@dz�?�A�B�
=?c�
@C33@   BB��                                     Bxp�h�  
�          @�=q�u@.�R�Ǯ���B�ff�u@A녿W
=�x��B�(�                                    Bxp�w�  T          @�G���=q@n�R�\)�=qB�Q쿊=q@p  >\@�  B�#�                                    Bxp��&  T          @�{?�{@Mp��z=q�D  B�?�{@�=q�@  ��B���                                    Bxp���  
�          @��
?Tz�@,(���G��hB���?Tz�@n{�o\)�1�HB���                                    Bxp��r  �          @�=q?�=q?�����H�
BKff?�=q@1G���G��Y\)B���                                    Bxp��  "          @��R>��
@�\��ff�0��B�
=>��
@�=�@@��B�Q�                                    Bxp���  T          @�(���
=@u@e�B&�
B�#׿�
=@5@��B\G�B�                                      Bxp��d  �          @��R��=q@��@R�\B�B��ÿ�=q@K�@�ffBJffB�                                    Bxp��
  �          @��׿��@���@6ffA��B�=���@^{@s�
B/�
B�8R                                    Bxp��  "          @��\����@�Q�@P  Bz�B�\����@E@�(�BC�RB��
                                    Bxp��V  "          @���	��@s33@_\)B��B����	��@5�@�=qBJQ�B�z�                                    Bxp�	�  �          @����\)@h��@J=qB=qB�Ǯ�\)@0��@|��B:��C�                                    Bxp��  �          @�\)���@H��@o\)B-�B��H���@Q�@�z�BWp�C
z�                                    Bxp�'H  �          @��ÿ�ff��@�  BnQ�Cv�f��ff�]p�@qG�B9  C}!H                                    Bxp�5�  �          @�{���
���
@��\B�{C?�����
���R@�33B�8RC`�                                    Bxp�D�  "          @�=q���?���@s33Bwp�C�{���>k�@~�RB�C+��                                    Bxp�S:  T          @���@@L(�BTffB���?�  @g
=B�8RCW
                                    Bxp�a�  �          @��Ϳ�{@E�@i��B=�B�3��{@@���Bo�\B�#�                                    Bxp�p�  �          @�=q�\(�@c�
@l��B5(�B�(��\(�@#33@��RBk��B�Q�                                    Bxp�,  �          @�\)>��@*=q@��Bj(�B��>��?�G�@�p�B���B��{                                    Bxp���  �          @�G����H@@��@���BI
=B��Ϳ��H?�
=@�z�By��B�(�                                    Bxp��x  �          @�  �W
=@qG�@XQ�B$��B�
=�W
=@5�@�ffB[�
B�                                      Bxp��  
�          @�녾�  @�@��Bw\)B��q��  ?��R@�Q�B��3B�Ǯ                                    Bxp���  T          @�\)�.{?��@��B���B{�.{<�@�
=B�G�C*{                                    Bxp��j  T          @�(�=��
?�Q�@�B��B��\=��
?=p�@�Q�B�.B�Q�                                    Bxp��  T          @�
=>�\)@G�@�G�B��fB�#�>�\)?J=q@�z�B��=B���                                    Bxp��  T          @�ff���@ ��@�\)Br��B�����?��@��RB���B�Ǯ                                    Bxp��\  �          @�G�?p��?�@�p�B�aHBrp�?p��>�
=@�B�� A�G�                                    Bxp�  "          @��
?�G�?�Q�@��B�ffB.
=?�G����
@�ffB��=C���                                    Bxp��  �          @�Q�?���?�=q@�{B�B�R?��׽���@��B��)C��q                                    Bxp� N  �          @��?���?�  @�=qB�z�B{?��;W
=@�p�B��HC���                                    Bxp�.�  �          @���?�(�?c�
@�33B�Q�A�?�(���=q@�B��C���                                    Bxp�=�  "          @�G�@0��?��@���BW��A��@0��<��
@�BbQ�>�{                                    Bxp�L@  T          @���?�  ?�p�@��
Brp�B3Q�?�  ?(�@��B��RA��R                                    Bxp�Z�  "          @���?p��?Tz�@��B��B%��?p�׾\@��
B��\C�                                    Bxp�i�  T          @�z��
�H@Q�@"�\BffC���
�H?���@?\)BCffC\)                                    Bxp�x2  �          @�{��(�@ �׿�
=�iC�=��(�@1녿E����\CL�                                    Bxp���  T          @�  ���\@,�Ϳ��
�v�HC�����\@@  �Q��33C(�                                    Bxp��~  
Z          @�(����
@��G����C8R���
@6ff��\)�b=qCG�                                    Bxp��$  "          @������\@
=��  ���HCh����\@�R��
=�<  C                                      Bxp���  
�          @������@p��<����{C������@7��z���C��                                    Bxp��p  
Z          @ȣ����@z��l(��p�C�\���@:=q�Dz�����C�R                                    Bxp��  �          @ə���ff@G
=��\����C�)��ff@e����H�W�C!H                                    Bxp�޼  �          @ʏ\���R@Q��G
=��(�CY����R@Dz�����G�C8R                                    Bxp��b  "          @�����@$z��G
=��{C(����@P  �������C8R                                    Bxp��  "          @�Q����H@P���,(���{C�����H@tz�����\CG�                                    Bxp�
�  
�          @�ff��G�@L(��H����33CQ���G�@w
=�������C	�R                                    Bxp�T  �          @ƸR����@\(������p�CE����@r�\�xQ��G�C�\                                    Bxp�'�  �          @�  ��(�@`  ��
��C���(�@}p������Lz�CxR                                    Bxp�6�  T          @�ff����@_\)��G���=qC�H����@tz�Y����(�Ck�                                    Bxp�EF  �          @��
���@hQ쿋��$  C�3���@q녾#�
���C�=                                    Bxp�S�  T          @�{���@l�Ϳ�G���Q�C8R���@��׿J=q���
C
�f                                    Bxp�b�  
�          @�����@w
=�����p��C�\����@�z������
C	��                                    Bxp�q8  �          @�p����H@X�ÿ�G����\C����H@n{�^�R�Ch�                                    Bxp��  �          @�{��\)@W����Tz�C�f��\)@g������=qC�                                    Bxp���  
�          @�����@X�ÿ�����p�C�{����@qG�������C��                                    Bxp��*  �          @�(���p�@XQ�� ����z�C�)��p�@x�ÿ����s�C
�H                                    Bxp���  �          @�33���@O\)������
C���@g
=���
�
=C&f                                    Bxp��v  
(          @�33��33@H�ÿ�33��G�C{��33@`�׿��� ��C)                                    Bxp��  �          @������@S�
� ������C=q����@l�Ϳ����)�C+�                                    Bxp���  �          @��H��  @QG��������Ck���  @h�ÿ��\��RC��                                    Bxp��h  T          @�=q���H@j�H����p�C�����H@��\��33�.ffC�\                                    Bxp��  �          @������
@n�R��=q����C�����
@�=q�\(���\C	
                                    Bxp��  �          @�(��p��@���(�����C^��p��@��׿O\)��C =q                                    Bxp�Z  �          @������@}p���
=���\C	!H����@�=q�c�
�G�C�                                    Bxp�!   �          @�z��|��@�z��p���G�C���|��@�녿�\)�(��C��                                    Bxp�/�  �          @�����Q�@j�H��33����C�)��Q�@��׿n{���C
=q                                    Bxp�>L  �          @�
=��=q@b�\�ٙ��}��C����=q@vff�E���33Ch�                                    Bxp�L�  
�          @Ǯ��\)@O\)��=q�33C:���\)@Z=q�u��RC�                                    Bxp�[�  �          @�����@n{�   ���C�\��@��H��G���
C
��                                    Bxp�j>  	�          @�=q���@u��   ���
CG����@��R�}p��Q�C	�)                                    Bxp�x�  	�          @ə�����@|(���\)��Q�C&f����@��ÿW
=����CǮ                                    Bxp���  �          @�Q���G�@x�ÿ�����33Cp���G�@�
=�L�����C	!H                                    Bxp��0  �          @�z����H@�ff��(��Up�CG����H@����
�9��C�)                                    Bxp���  �          @˅���@�G������i��C�{���@������H��z�C)                                    Bxp��|  �          @�z����
@�p������G�
C	�����
@�zᾙ���+�C}q                                    Bxp��"  �          @θR��G�@��Ϳ�Q��)��C���G�@�=q�����
C	�q                                    Bxp���  �          @�ff��=q@��H��G��4��C����=q@��þL�Ϳ���C
h�                                    Bxp��n  �          @�\)��33@��c�
����CJ=��33@���>8Q�?���C
�                                    Bxp��  �          @�Q����@��\��{�@  C5����@���?O\)@��C�
                                    Bxp���  �          @У����\@�(����R�+�C�����\@��?\(�@�\C�                                    Bxp�`  �          @�Q����@�Q�����33C�f���@�Q�?
=@�\)C�                                    Bxp�  �          @�  ���
@��ÿ\)��{C�\���
@���?
=@��RC�
                                    Bxp�(�  �          @У����H@��H����dz�C�����H@���?B�\@�p�CB�                                    Bxp�7R  �          @�  ��=q@�  �\(���\C
�
��=q@��\>u@C
\                                    Bxp�E�  �          @�  ���
@�Q쿽p��S�C	T{���
@����Q��N{C��                                    Bxp�T�  �          @љ���
=@�=q���H�*=qC����
=@�  �����ffC�                                     Bxp�cD  T          @�=q��z�@}p����
�Q�Cp���z�@��H�#�
�L��C��                                    Bxp�q�  �          @������@�녿��\��C=q����@�<�>uC^�                                    Bxp���  �          @љ����
@x�ÿ��R�.�HC�\���
@�=q�k���p�C��                                    Bxp��6  �          @������H@|(���\)�G�CJ=���H@��H��Q�J=qC8R                                    Bxp���  �          @У���33@y����\)��RC�\��33@�녽��Ϳk�CxR                                    Bxp���  �          @�������@y�������D  C8R����@��
��Q��HQ�C��                                    Bxp��(  �          @љ���
=@r�\��\)��
=C����
=@�z�Y����  C5�                                    Bxp���  �          @�G�����@n�R��\)��33Cc�����@��H�^�R����C�f                                    Bxp��t  �          @�G����@���{�@z�C�\���@�(��B�\�ٙ�C��                                    Bxp��  �          @����ff@�z����p�Ch���ff@��R�.{��ffC^�                                    Bxp���  
�          @У����@����G����HC	J=���@��H��33�"{Ch�                                    Bxp�f  �          @�����\@����(Q����C=q���\@���  �S�C�=                                    Bxp�  �          @�33��ff@�(��#�
���RC����ff@�(���{�?\)C��                                    Bxp�!�  �          @�33���@�=q�\)����C�
���@�����  �/33C                                     Bxp�0X  �          @Ӆ�s�
@��\�����z�C =q�s�
@��ÿ������B��{                                    Bxp�>�  �          @����Y��@��\�.{�ř�B�ff�Y��@����33�G\)B��H                                    Bxp�M�  �          @Ӆ����@x���=p����C
�����@��׿����z�CB�                                    Bxp�\J  �          @��
����@Mp��a���C������@�  �&ff��=qC
}q                                    Bxp�j�  �          @�G���z�@+���G����C�\��z�@g��N{��Q�Cu�                                    Bxp�y�  �          @љ���(�@���{�C!H��(�@��=�G�?�G�CY�                                    Bxp��<  �          @�Q���\)@a��'����CO\��\)@��\����j�HC=q                                    Bxp���  �          @ƸR�z�H@�������C^��z�H@��R?�33A/33CG�                                    Bxp���  �          @��H�x��@~�R�L���ffC\)�x��@y��?Q�A\)C�                                    Bxp��.  T          @��@j=q����@�
=BE�RC�h�@j=q�\)@���B-
=C�Ff                                    Bxp���  �          @Å@3�
���@�Bu�C��@3�
��(�@���B`p�C�E                                    Bxp��z  �          @Å?�Q�>��
@�=qB�
=A=q?�Q쿏\)@��RB�ffC�                                    Bxp��   �          @�G�?�p�?�=q@��B�� B;��?�p�>L��@���B��@�(�                                    Bxp���  �          @�p�@   ��@���B�=qC���@   �33@��RB}\)C�&f                                    Bxp��l  �          @�p�?��#�
@��
B��C�z�?���p�@�(�B��
C�+�                                    Bxp�  �          @�ff?z�H?��@�=qB��fA��
?z�H����@�  B��C�ٚ                                    Bxp��  �          @�
=>�\)?+�@�(�B�B���>�\)�}p�@ʏ\B�z�C��                                    Bxp�)^  �          @�ff?�
=�.{@��B��\C���?�
=��Q�@�p�B�B�C��                                    Bxp�8  �          @ə�?�zᾳ33@�z�B���C���?�z��@��HB��)C���                                    Bxp�F�  �          @˅?h��=���@���B��@�z�?h�ÿ��R@�
=B�ffC��f                                    Bxp�UP  �          @�(�>��?�
=@��B��3B�z�>��>���@�33B���B 33                                    Bxp�c�  �          @��H    ?�\)@�p�B�B�Ǯ    ��\)@��B���C�B�                                    Bxp�r�  T          @�G����R?�{@�G�B���B����R�#�
@�Q�B�B�C6{                                    Bxp��B  �          @�Q�=�\)@�\@��B���B�  =�\)>�G�@�{B��B���                                    Bxp���  �          @�z�@ ��>�G�@��B�Q�A�R@ �׿�  @�p�B~{C�3                                    Bxp���  �          @�{?��?��@��HB�� B	p�?���   @�p�B�Q�C��                                    Bxp��4  �          @ə�@%?
=q@�B�\A<��@%�u@�(�Bp�C���                                    Bxp���  �          @�Q�?��R@G�@�  Bj33B��=?��R?�=q@��
B���BO33                                    Bxp�ʀ  �          @θR?���@333@��Bu��B���?���?��R@�z�B��{B7��                                    Bxp��&  �          @Ϯ?Ǯ@!�@��Bzz�Biff?Ǯ?p��@�ffB�=qA�
=                                    Bxp���  �          @���?8Q�@P  @�G�Bk�B�(�?8Q�?�Q�@�ffB�  B�\                                    Bxp��r  �          @У׿=p�@��@��B8�RB�(��=p�@A�@���BtG�Bϊ=                                    Bxp�  �          @ʏ\�p��@�\)@W
=B�B�B��p��@�p�@��B<��B�G�                                    Bxp��  �          @�33���@��@Z=qB�B�  ���@��@��B=z�Bнq                                    Bxp�"d  T          @��
���@��H@j=qBffB˔{���@|��@��
BGB��                                    Bxp�1
  �          @���aG�@��R@O\)A�G�B�LͿaG�@��@�=qB5�RBʊ=                                    Bxp�?�  �          @��ÿ.{@���@P��A�{B£׿.{@�\)@�G�B:�\B�8R                                    Bxp�NV  �          @�Q��@�  @S33B
=B���@}p�@�  BA��B�\                                    Bxp�\�  �          @�ff��@�  @g�Bz�B�����@XQ�@�BXB��H                                    Bxp�k�  �          @�=q��  @�33@hQ�B�B�Q��  @?\)@��HBS��B�                                    Bxp�zH  �          @���7
=@�{@@  A��RB��3�7
=@O\)@�  B+{Cn                                    Bxp���  �          @�\)�%�@�Q�@N�RB��B�u��%�@P  @�  B6�C xR                                    Bxp���  �          @��
�C�
@��R@L��A��\B����C�
@Mp�@�ffB-�
C�)                                    Bxp��:  �          @�p��  @�@5�A�\B�aH�  @p��@~{B(�RB��
                                    Bxp���  �          @����   @�p�@!G�A���B�.�   @�=q@n�RB�HB�=q                                    Bxp�Æ  �          @�
=�Q�@�p�@$z�A�p�B�ff�Q�@U�@eB%  B�                                    Bxp��,  T          @��H�h��@H��@l��B\)C0��h��@�\@���B:=qC�R                                    Bxp���  �          @��H�g
=@W
=@c�
BQ�C	
=�g
=@�\@��\B5�C��                                    Bxp��x  �          @�Q��tz�@XQ�@L(�A��C
p��tz�@=q@~�RB%�C                                    Bxp��  �          @���^�R@_\)@b�\B�C���^�R@�H@��HB6�HC8R                                    Bxp��  �          @Å�r�\@e@H��A�{C}q�r�\@'�@\)B#�\CO\                                    Bxp�j  �          @�=q��
@�{@J�HA�  B�p���
@j�H@�=qB6�B�3                                    Bxp�*  �          @����@�Q�@4z�A��B�ff�@u�@�  B&��B���                                    Bxp�8�  �          @�  �y��@~�R@
�HA��
Ck��y��@P  @J=qA�
=C33                                    Bxp�G\  �          @�����@A�@=p�A�{C�����@
=@j�HB��C\)                                    Bxp�V  �          @�{��=q@g�@0  A�  C����=q@/\)@g�B�C�                                    Bxp�d�  �          @��
��
=@fff@%�A��
Cz���
=@1G�@]p�B	�HC��                                    Bxp�sN  
�          @��H���@�  ?�G�A^ffC\���@\(�@"�\A�ffCB�                                    Bxp���  �          @�33��Q�@��
?
=q@�p�C}q��Q�@�G�?�A�p�Cz�                                    Bxp���  �          @�33��G�@�(�>�\)@$z�C�\��G�@�(�?��
A`Q�Ch�                                    Bxp��@  �          @ʏ\��Q�@��>�@���C����Q�@��?�Atz�C	��                                    Bxp���  �          @ȣ���ff@~{�p���
�RC�
��ff@��\>8Q�?�33C
                                    Bxp���  �          @���=q@�녿J=q���C�q��=q@�33>�@�
=Cu�                                    Bxp��2  �          @�  ��p�@�33������C0���p�@���?O\)@�
=C}q                                    Bxp���  �          @�G����@�G��
=q��p�C����@�Q�?8Q�@��
C��                                    Bxp��~  �          @ə�����@��ÿh�����C������@��
>���@.{C                                      Bxp��$  �          @�G���Q�@�����\�p�C� ��Q�@��>B�\?�Q�C�R                                    Bxp��  �          @�G����@����  �=qCG����@�
=>u@(�C�
                                    Bxp�p  �          @ƸR��Q�@p�׿����HQ�C�f��Q�@\)������C                                    Bxp�#  �          @�����Q�@6ff��\)�{
=CY���Q�@L(��E���z�C��                                    Bxp�1�  �          @�(����R@fff��Q��YC�)���R@w
=�Ǯ�hQ�C�3                                    Bxp�@b  �          @\��=q@Tz����lQ�CxR��=q@g
=�����
C#�                                    Bxp�O  �          @��
��ff?�녿^�R��C!���ff@33���R�<(�C 0�                                    Bxp�]�  �          @������@S�
��Q�Y��C������@Mp�?Tz�AC�\                                    Bxp�lT  �          @�  �vff@��R�aG���C�\�vff@�33?��
A ��C��                                    Bxp�z�  �          @�
=���@Vff?��A��HC}q���@0��@   A�33C�                                    Bxp���  �          @�  ��{@\)@EA��C����{?��
@j=qB�C �R                                    Bxp��F  �          @�\)�o\)@�Q��%����
CL��o\)@�녿���?\)B�                                    Bxp���  �          @�\)�Mp�@��H�O\)��{B����Mp�@�=q��=q��
=B�Q�                                    Bxp���  �          @�Q��K�@��\�Tz���  B���K�@��\��33��z�B��                                    Bxp��8  �          @�G��H��@��\�E���
=B�
=�H��@�  �����c\)B�p�                                    Bxp���  �          @�(��;�@���Mp���=qB��
�;�@�{��
=�k�B�z�                                    Bxp��  �          @�33��
@���O\)��{B� ��
@�{����f�\B�k�                                    Bxp��*  �          @љ��33@��R�fff�\)B�q�33@�����
��ffB�p�                                    Bxp���  �          @�ff�G
=@�Q��XQ����
B�(��G
=@��ÿ�p����B���                                    Bxp�v  �          @�  �w
=@��J�H����C�R�w
=@��Ϳ�{���RC 33                                    Bxp�  �          @�  ��33@�G��>{��Q�Cc���33@�
=���H�t(�C��                                    Bxp�*�  �          @�  �Y��@��H�Y����33C ��Y��@�(�����  B�                                      Bxp�9h  �          @Ϯ�	��@�{��(���HB�3�	��@��,������B�8R                                    Bxp�H  �          @�33��Q�@����4z��ӅC���Q�@��ͿǮ�dz�C�q                                    Bxp�V�  �          @�(��~{@o\)�R�\��z�C�3�~{@�������HCJ=                                    Bxp�eZ  �          @�(��h��@��H�\)��=qC aH�h��@����=q��
B��)                                    Bxp�t   �          @����S�
@�  �<(���Q�B��{�S�
@������a�B�\)                                    Bxp���  �          @�{�333@����y����\B�� �333@�ff�!G����\B�                                    Bxp��L  �          @�{��@�z���p��$B�{��@����.�R���HBٸR                                    Bxp���  �          @�{�Ǯ@��>{��ffB�uÿǮ@�=q��{�L  B���                                    Bxp���  �          @����
@��,���ƣ�B�uÿ��
@�
=�s33�  B�Q�                                    Bxp��>  �          @ʏ\����@��R�.{�̸RBԮ����@�Q쿅��\)B���                                    Bxp���  �          @�Q���@�(��'
=�ƣ�B�����@�p����\��RB�                                    Bxp�ڊ  �          @�  �5@����H���HB�ff�5@��þ\�`��B�W
                                    Bxp��0  �          @�=q��@�
=�	�����\B�(���@��
�
=q���B��                                    Bxp���  �          @�=q�+�@���o\)�ffB�G��+�@��
�����
=B�
=                                    Bxp�|  �          @�  �k�@�������4=qB��k�@���<���ᙚB���                                    Bxp�"  �          @�ff���@�(���33�2z�B�(����@�ff�8����ffB�Ǯ                                    Bxp�#�  �          @�
=>8Q�@3�
��G��|G�B��R>8Q�@�����\)�8��B�=q                                    Bxp�2n  �          @ə�?5@����p�B��{?5@|(������HB��\                                    Bxp�A  �          @ȣ�?�ff?��H��Q�� Ba�?�ff@c�
��{�Q�HB�Ǯ                                    Bxp�O�  �          @���?��@<(�����_��B��?��@�33�W
=�\)B��
                                    Bxp�^`  �          @��
�333@����!�����B��f�333@ȣ׿8Q��У�B��
                                    Bxp�m  �          @�(��z�H@��
�2�\�иRB�Ǯ�z�H@�{���\��
B���                                    Bxp�{�  �          @��H�5@���#�
��
=B�#׿5@Ǯ�B�\��z�B�                                      Bxp��R  �          @ʏ\���R@��R�)���ƸRB�.���R@Ǯ�Y����\)B���                                    Bxp���  �          @�33�E�@�  ������B³3�E�@�G����
�L��B�
=                                    Bxp���  �          @��H��  @�p��\)��\)B��H��  @�=q�Ǯ�_\)B��{                                    Bxp��D  �          @�z��\@����33��Q�B��q��\@�33�8Q��\)B�8R                                    Bxp���  �          @�z�   @����������\B���   @�33��\)��RB�{                                    Bxp�Ӑ  �          @ə�>B�\@���#�
���B�33>B�\@�  �:�H�ָRB��                                     Bxp��6  �          @��>�\)@�
=��p���z�B��>�\)@��ý�G���G�B���                                    Bxp���  �          @�녿�@��Ϳ�G��8Q�B���@�  ?!G�@���B���                                    Bxp���  �          @ʏ\���R@�녾�z��*=qB������R@Å?���Ak33B���                                    Bxp�(  �          @��H��z�@��>���@B�\B�.��z�@�p�@p�A�p�B��                                    Bxp��  �          @�=q��Q�@��\�����B�Q��Q�@��R��33�L(�B�\                                    Bxp�+t  �          @�Q쿐��@��R��  �^ffB�ff����@�(�>�Q�@Q�B���                                    Bxp�:  T          @�  ��=q@��׿�Q��0z�B�aH��=q@�33?+�@�ffB��                                    Bxp�H�  �          @ȣ׿���@����O\)��{B�33����@���?�ffA�RB�W
                                    Bxp�Wf  �          @��ÿ��@�zᾳ33�O\)B�\���@��R?\A`��BȨ�                                    Bxp�f  �          @�  ��
=@��ÿ���f{B�W
��
=@�\)>�\)@#33B�W
                                    Bxp�t�  �          @��ÿ�@�
=�P����B�33��@�
=�����m�B�\                                    Bxp��X  �          @ə����@���#33��z�B�#׿��@�=q�=p���=qB�                                    Bxp���  �          @�33��\)@�������33B��Ϳ�\)@�
=�Ǯ�`  B�aH                                    Bxp���  �          @ʏ\���@�
=�޸R��  BȽq���@ƸR>#�
?���B��                                    Bxp��J  �          @�G���{@��׿�{�G�
B��쿎{@���?
=q@�\)B�ff                                    Bxp���  �          @ə���{@�=q�p���	�B�=q��{@�=q?uA��B�=q                                    Bxp�̖  �          @��ÿУ�@�=q�
=���BԨ��У�@�ff�����2�\BҮ                                    Bxp��<  �          @Ǯ���R@�
=��Q��VffB�W
���R@�z�>\@`  B�Q�                                    Bxp���  �          @�\)����@�����\��
B�LͿ���@�{?W
=@���B�(�                                    Bxp���  
�          @�Q쿏\)@�z�8Q��33Bȳ3��\)@���?�(�A�BɊ=                                    Bxp�.  �          @�\)��Q�@�33=���?uB���Q�@���?�p�A�z�B�=q                                    Bxp��  �          @�  ��33@��H�����\)B�aH��33@�\)?��AC
=B�Ǯ                                    Bxp�$z  �          @Ǯ��
=@�G����R�9��Bή��
=@��H?���Aj{BϏ\                                    Bxp�3   �          @Ǯ���R@�G�=L��>�G�B����R@�\)?�A���B�8R                                    Bxp�A�  �          @Ǯ��Q�@��׾L�Ϳ��B�𤿸Q�@���?�A{
=B�                                      Bxp�Pl  �          @Ǯ��@��R�]p��	p�B޸R��@���������ffBس3                                    Bxp�_  �          @�
=��{@����B�\��\B��
��{@��?ٙ�A��\Bɮ                                    Bxp�m�  �          @�  �E�@�=q?��A�HB�ff�E�@�{@8��Aݙ�B�
=                                    Bxp�|^  �          @�G��333@�Q�?�=qAip�B�33�333@�
=@XQ�B�B�33                                    Bxp��  �          @�G���G�@Å?��AQ�B��쿁G�@�\)@9��A�  B��                                    Bxp���  �          @��þ\@��
?��\A
=B�\�\@�  @8��A�p�B��)                                    Bxp��P  �          @ƸR���
@�zᾅ��(�B�𤾣�
@��?�
=A}��B�(�                                    Bxp���  �          @�
=�xQ�@Å>#�
?�G�B��ÿxQ�@�  @�A�
=B�\                                    Bxp�Ŝ  �          @Ǯ��  @�z������B�uÿ�  @��?�A���B�L�                                    Bxp��B  T          @Ǯ�(�@���>�z�@(��B�W
�(�@��@p�A�B�#�                                    Bxp���  �          @�\)��@���?=p�@�z�B��Ϳ�@�  @'�A��B��f                                    Bxp��  �          @ƸR�z�H@�=q��\��
=B�aH�z�H@��?�Q�AXz�B��
                                    Bxp� 4  �          @�ff��  @�\)��{�L(�Bˏ\��  @���?�=qAo
=B�\)                                    Bxp��  �          @�z��@�=q�O\)��z�B�\��@���?�=qA#\)B�L�                                    Bxp��  �          @Å���H@���>��@!�B�.���H@���@33A�ffBܳ3                                    Bxp�,&  �          @��
�Q�@��?ٙ�A��B����Q�@�G�@Tz�B��B�B�                                    Bxp�:�  �          @�(���=q@���?��A{Bң׿�=q@�z�@5Aޏ\B�=q                                    Bxp�Ir  �          @�G����H@��R��(�����B�k����H@���?�A\  B�W
                                    Bxp�X  �          @��
��@��H���� ��B���@�(�?J=q@�G�B�u�                                    Bxp�f�  �          @ȣ��K�@�  �
=��p�B����K�@�Q�5���B�ff                                    Bxp�ud  �          @�Q��<(�@�33�'��ƣ�B�\�<(�@��k���B��f                                    Bxp��
  �          @ȣ��-p�@�33�8���ۅB�\)�-p�@��ÿ�z��+\)B�B�                                    Bxp���  �          @����5@�(��G�����B����5@�zΌ���V�RB�z�                                    Bxp��V  �          @�\)���@����ff��\)B�q���@�p���\)�$z�B�3                                    Bxp���  �          @�{�\)@�p���R���
B�k��\)@��
��(���  B���                                    Bxp���  �          @Ǯ�7�@��
�%��G�B���7�@�ff�^�R� z�B�z�                                    Bxp��H  �          @ȣ��8Q�@�p��W���B�B��8Q�@��ÿ޸R��G�B�W
                                    Bxp���  �          @����4z�@�������=qB�.�4z�@�z�8Q��ٙ�B�(�                                    Bxp��  �          @��H��@�Q�����B����H��@�Q�#�
��  B��                                    Bxp��:  �          @����U�@|���C�
��
=C(��U�@�\)�����v�HB�G�                                    Bxp��  �          @�Q��N{@c�
�e���
C#��N{@�G��p����B��q                                    Bxp��  T          @��x��@)���\)�!33C�q�x��@s�
�8����{C�\                                    Bxp�%,  �          @�=q�q�@C33�hQ��33C{�q�@��\������C��                                    Bxp�3�  �          @�ff�S33@8����  �)\)C
�q�S33@����4z����HC&f                                    Bxp�Bx  �          @�  �HQ�@"�\�����3��C���HQ�@n{�=p���Q�C�                                    Bxp�Q  �          @��H�7
=@���  �=\)C��7
=@aG��@  �  C                                    Bxp�_�  �          @���<(�?�=q��33�KG�C�<(�@6ff�U���C��                                    Bxp�nj  �          @��H� ��@��ÿW
=���Bߣ�� ��@���?aG�A  B߳3                                    Bxp�}  �          @�33�0��@�\)������\)B��0��@�=q�\)��Q�B�8R                                    Bxp���  �          @��\)@�p������@��B���\)@���?
=@�(�B��                                    Bxp��\  �          @��\�޸R@�ff������B�W
�޸R@���?�ffA�(�B�G�                                    Bxp��  �          @��þ�
=@�\)��Q��n�RB���
=@���?�  A|Q�B�{                                    Bxp���  �          @�G���\@�  �#�
��
=B��\��\@�\)?�Q�A���B��                                    Bxp��N  �          @�\)�.{@��!G����HB��=�.{@��\?�p�APz�B���                                    Bxp���  T          @�33�}p�@�
=����)��Bȏ\�}p�@��?���A���B�p�                                    Bxp��  �          @�G��z�H@��>��@�G�BȊ=�z�H@�ff@p�A�\)B�aH                                    Bxp��@  �          @���=u@���?�\@��B��H=u@�Q�@�A�z�B�                                    Bxp� �  �          @�=q�
=@�p�?Q�A�HB��
�
=@���@,(�A���B�B�                                    Bxp��  �          @�녿333@��R?�{A���B¸R�333@��H@W�B��B�z�                                    Bxp�2  �          @���=�\)@��?��RApz�B��\=�\)@�
=@R�\B�\B�Q�                                    Bxp�,�  �          @�\)    @�p�@�HA�ffB�    @�G�@���B3��B�                                    Bxp�;~  �          @�ff?�{@a�@�=qB6ffBx��?�{?�
=@���Bw�
B7�H                                    Bxp�J$  T          @�
=?G�@���@x��B-�B�G�?G�@�R@�G�BzB�8R                                    Bxp�X�  T          @���?�z�@p��@�=qB5z�B���?�z�@��@��B}ffBbG�                                    Bxp�gp  �          @�  @(�@s�
@g
=Bz�BpQ�@(�@@�ffB^��B;�\                                    Bxp�v  �          @�ff@ ��@���@.{A���B��@ ��@L��@�=qB<�Bgz�                                    Bxp���  
�          @�z�@.{@J�H@b�\B!(�BE�\@.{?�G�@�p�BW  B��                                    Bxp��b  �          @�33���
@��\?�
=APz�B�zἣ�
@��@7
=B��B��\                                    Bxp��  �          @����e�@�Q�ٙ����C z��e�@�=q�#�
�\B�G�                                    Bxp���  �          @�z��~{@��׿�ff���C���~{@�(��W
=�G�C�)                                    Bxp��T  �          @��
���
@l(��'�����C
!H���
@��
��
=�1�CL�                                    Bxp���  �          @��H����@1G��I������C�\����@l(����R����C�                                    Bxp�ܠ  �          @������@9���<(����Cn����@o\)�޸R��  C�                                    Bxp��F  �          @����{@R�\�,����=qCs3��{@��ÿ����O�
C	��                                    Bxp���  �          @�=q�r�\@�(���\)�y�C���r�\@�p�<#�
=���C{                                    Bxp��  T          @�����
@r�\�z�����C	c����
@��׿�R��ffC�                                    Bxp�8  �          @�\)�n{@�ff�����V�\C�H�n{@��>��@"�\C �H                                    Bxp�%�  T          @��ÿY��@��ÿ0����{B�{�Y��@�(�?�(�A^�\B�u�                                    Bxp�4�  �          @�z�fff@ȣ׿L����
=B�aH�fff@���?���AR�\Bĳ3                                    Bxp�C*  T          @�p����@��������g33B��H���@�\)?#�
@��RB�33                                    Bxp�Q�  �          @θR�˅@��
�
=q��Q�B�B��˅@�  =L��>�B�z�                                    Bxp�`v  �          @��p�@�z��*�H�ď\B����p�@���
=���B܏\                                    Bxp�o  �          @��33@��\�H����33B�R�33@������$  B��
                                    Bxp�}�  �          @�ff��@����j=q���B�.��@��Ϳ�\����B���                                    Bxp��h  �          @�p���
=@��
�������B�B���
=@�  �33��z�B׽q                                    Bxp��  �          @��z�@AG���
=�\p�B���z�@���������B��                                    Bxp���  �          @ҏ\�5�@Z�H���\�=��C�f�5�@���R�\����B��f                                    Bxp��Z  �          @�{�+�@7�����P�
C���+�@���l���33B�{                                    Bxp��   �          @��
=q@����L(���RB�{�
=q@�녿�  �
=B��q                                    Bxp�զ  �          @����p�@QG���\)�^{B����p�@��mp��
=B�ff                                    Bxp��L  �          @Ϯ@\��@p  �s�
�Q�B=z�@\��@��
�p�����BZ�R                                    Bxp���  T          @�ff?�@{�����{�HBI�\?�@����
=�3Q�B�Ǯ                                    Bxp��  T          @ə��333@L�������c�B̮�333@���c33��BĔ{                                    Bxp�>  �          @ə����@:�H����a�B�Ǯ���@���u��Q�B�u�                                    Bxp��  �          @�33��
@O\)���R�K�B�����
@�\)�Mp����B�{                                    Bxp�-�  �          @�p���z�@j=q��\)�EB�{��z�@����6ff��33B��)                                    Bxp�<0  �          @�zῙ��@hQ��hQ��.{Bب�����@�ff�z���ffBг3                                    Bxp�J�  �          @q��{@33����p�C��{@3�
�aG��\Q�C Y�                                    Bxp�Y|  �          @<�Ϳ���G�?�Q�B��C^
����
=?J=qA�{Ce#�                                    Bxp�h"  �          @��R�K�>�G�?�  A��
C,��K���?ǮA�=qC6h�                                    Bxp�v�  �          @��R���R@��?O\)A�CO\���R?�Q�?У�A�
=CG�                                    Bxp��n  �          @�ff�i��@A녿Tz��ffC@ �i��@HQ�>�{@��Cff                                    Bxp��  �          @����(�@AG��fff�ffCaH��(�@H��>�=q@6ffCQ�                                    Bxp���  �          @��\����@HQ�5��33Cn����@J�H>��H@�G�C�                                    Bxp��`  �          @�ff��p�@:�H>��R@Tz�C����p�@'
=?�{Am�C�                                     Bxp��  T          @����  @   ?�{AS33C�R��  ?�?���A�  Cc�                                    Bxp�ά  �          @�(����
@��?(�@�p�C����
@ ��?���A��HC��                                    Bxp��R  �          @�����\?���z��FffCY����\?��>�G�@�33C��                                    Bxp���  �          @����G�@-p������7�C#���G�@:=q<#�
=�Q�CE                                    Bxp���  �          @������H@;�����Q�C�q���H@;�?\)@���C��                                    Bxp�	D  �          @�G���@
=�������C:���@��>�p�@~�RC�H                                    Bxp��  �          @�(���
=@>{��G���z�Cs3��
=@:=q?333@�C��                                    Bxp�&�  �          @���x��@Mp�>�\)@J�HC}q�x��@7�?��HA�=qC��                                    Bxp�56  �          @�����G�@R�\>�{@mp�CǮ��G�@;�?�ffA��
C
=                                    Bxp�C�  �          @�(��Z=q@`  �����HC:��Z=q@R�\?���Ab�HC�q                                    Bxp�R�  �          @�z�8Q�@�ff?p��A"�RB��)�8Q�@�
=@5�A��B�k�                                    Bxp�a(  �          @��\���@�p���\)�HQ�B�(����@�z�?�
=A��\B��H                                    Bxp�o�  T          @�녿޸R@���?0��@���B�{�޸R@���@�RA�{B�=q                                    Bxp�~t  �          @�Q�=��
��녿:�H�
�HC�xR=��
���   ���HC���                                    Bxp��  �          @�{�\)�l(������Cn�R�\)�N�R����=qCkJ=                                    Bxp���  �          @����\)��(��u�:�RCC5���\)�B�\������=qC=�                                    Bxp��f  �          @�����ff�(��У�����C;���ff=L�Ϳ޸R���HC3aH                                    Bxp��  �          @������
?�(��У�����C"^����
?�p��s33�+\)C��                                    Bxp�ǲ  �          @���z=q@>�R������C�q�z=q@_\)�#�
��  C
O\                                    Bxp��X  �          @�p���=q@g
=�\�z=qC
h���=q@z�H�����RC�                                    Bxp���  �          @�(����H@Vff���`  C�H���H@hQ����z�C�)                                    Bxp��  �          @�p���=q@0��� ����G�C33��=q@Tz�L������Cp�                                    Bxp�J  �          @�(���{@(������Q�C�H��{@Vff��
=�9�Cz�                                    Bxp��  �          @�  ��Q�@-p��-p���C�\��Q�@b�\���H�lQ�CE                                    Bxp��  �          @�(��N�R@�
=��=q���
B���N�R@�33�u�&ffB�=q                                    Bxp�.<  �          @���U�@�(������  C ٚ�U�@�G��\)��
=B��                                    Bxp�<�  �          @�ff�s�
@fff�p�����C���s�
@�p��&ff����CaH                                    Bxp�K�  �          @�z��g
=@a��\)���C���g
=@�
=�n{�(�C��                                    Bxp�Z.  �          @�33�Z�H@G��@���=qC	���Z�H@�G�������(�CG�                                    Bxp�h�  �          @�(���p�@<(���(��I��B���p�@�=q�*=q��z�B߽q                                    Bxp�wz  �          @�=q�'�@;��z�H�3�C���'�@�\)�{��p�B�                                    Bxp��   �          @��\�\)@;���
=�I�RB��q�\)@�{�>{����B�=                                    Bxp���  �          @����,(�@W��q��$��C ���,(�@��\�
=q����B��                                    Bxp��l  �          @����:=q@Q��p���"��C���:=q@�\)����B��H                                    Bxp��  �          @������@XQ�� ���Ə\Ch����@��H�z�H��C                                      Bxp���  �          @�
=��33@33�S�
��HC���33@Z=q�	�����
Cٚ                                    Bxp��^  �          @�z���33?��
��
=��G�C$���33@
=q��
=�8��C��                                    Bxp��  �          @����z�@   �Mp��Q�C�H��z�@c�
��(����\CO\                                    Bxp��  �          @�p���Q�?�p��������CaH��Q�@"�\�fff��C=q                                    Bxp��P  �          @�ff��ff?�����bffC ����ff@p��\)��C��                                    Bxp�	�  �          @�z�����@\)����p�C�����@ff?E�@�C��                                    Bxp��  T          @�(���{@ff��\)�1C����{@&ff���ͿuCff                                    Bxp�'B  �          @���6ff@�G���(����RB����6ff@�G�?�{A���B�p�                                    Bxp�5�  �          @�33�K�@�(��ٙ���(�B�{�K�@��>\@e�B�B�                                    Bxp�D�  �          @Å�L(�@�����
��ffB���L(�@���=q�#33B�33                                    Bxp�S4  �          @�ff��\@���=u?�B��H��\@�ff@ffA�\)B��H                                    Bxp�a�  �          @����33@��H�G���ffB�(��33@�{?\AjffB�8R                                    Bxp�p�  �          @����*=q@�ff�aG��	��B���*=q@�33?��A�  B�aH                                    Bxp�&  �          @�\)�p�@�녿��
�o
=B�33�p�@�\)?8Q�@�ffB��                                    Bxp���  T          @��H�z�@�
=�=p�����B�#׿z�@���?�(�A�{B��=                                    Bxp��r  �          @�(�>�G�@�\)�0����G�B���>�G�@�  ?�\A��RB�Q�                                    Bxp��  �          @�(��h��@��������B�LͿh��@�Q�>�Q�@W�B�33                                    Bxp���  �          @�33��Q�@�(��������B��ÿ�Q�@��H=���?h��B�aH                                    Bxp��d  �          @Ǯ��H@����������B�G���H@�\)=u?�B�R                                    Bxp��
  �          @�����z�@6ff��G�����CaH��z�@*=q?�ffA1C0�                                    Bxp��  �          @����{�:�H?xQ�A(��C<���{���?
=@��C?�
                                    Bxp��V  �          @�����
>�33>.{?�(�C0u����
>�  >���@@��C1xR                                    Bxp��  �          @����Q�?�=q>k�@�C �{��Q�?�=q?s33A��C#L�                                    Bxp��  �          @��R���@H�ÿ+���(�C�R���@H��?(��@�G�C�3                                    Bxp� H  �          @����G�@g
=��G��J�RC���G�@s�
>��
@L��C
ff                                    Bxp�.�  �          @�ff�u@��?   @��C��u@c33@Q�A�p�C	L�                                    Bxp�=�  T          @�=q���@e�?xQ�A (�C@ ���@7�@ffA�p�C\)                                    Bxp�L:  T          @����(�@h��<��
>8Q�C����(�@S�
?�G�Amp�Ch�                                    Bxp�Z�  �          @�Q�����@c�
����ffCz�����@]p�?}p�A�CE                                    Bxp�i�  �          @������@{��#�
����C����@i��?�(�AYC�                                    Bxp�x,  �          @ə���Q�@?\)�(����\C� ��Q�@>�R?+�@�z�C�)                                    Bxp���  �          @�����@C33�����  C�q���@@  ?B�\@�(�CW
                                    Bxp��x  �          @��H���@��33�H��CB����@G�?&ff@���C�\                                    Bxp��  �          @�
=���
@XQ�   ��(�C�����
@Q�?s33Ap�CQ�                                    Bxp���  �          @�Q���(�@:�H��G��9C����(�@K�=u?
=Cn                                    Bxp��j  T          @�33��
=?��
��\)�m�C'�)��
=?��p���	G�C"�)                                    Bxp��  �          @�{��ff?�
=�˅�r�HC%���ff?��H�\(��p�C!�                                    Bxp�޶  �          @�p���{@"�\����33C����{@Dz��R���\Cc�                                    Bxp��\  T          @�(���ff@����1�C���ff@,�ͽL�Ϳ   C��                                    Bxp��  �          @������@}p��(���
=C	O\����@w
=?���A,z�C

                                    Bxp�
�  �          @��H�l(�@��R?5@أ�C \�l(�@~�R@'
=AͮC��                                    Bxp�N  �          @�p��W�@���?У�Au�B���W�@p��@a�B�C�)                                    Bxp�'�  �          @�z��J�H@�\)?\Ag
=B���J�H@x��@^{B
�RC33                                    Bxp�6�  �          @\�6ff@��@�RA�\)B��
�6ff@L(�@��RB233C�                                    Bxp�E@  �          @�G��;�@��\@�
A�=qB�k��;�@aG�@y��B!��CǮ                                    Bxp�S�  �          @�\)�K�@�G�?�  AiB�{�K�@n{@XQ�B
��C��                                    Bxp�b�  �          @�����ff@dz�?�33A�p�CJ=��ff@�@N�RB�CW
                                    Bxp�q2  �          @����z�H@���@
=A�33CB��z�H@0  @g
=B=qC�                                    Bxp��  �          @���^�R@��@�HA�{C�)�^�R@0  @|��B&�
C��                                    Bxp��~  �          @�{�W�@���@,��Aأ�C�q�W�@!�@��B2�C)                                    Bxp��$  �          @�{�^�R@qG�@7
=A��HC���^�R@��@�{B5��C�f                                    Bxp���  �          @�
=��
=@C33@8��A�  C(���
=?��
@z=qB$(�C 
=                                    Bxp��p  �          @�\)��  @>{?��HA��C����  ?��H@3�
A�G�C�=                                    Bxp��  
�          @�����?�@�HA��
C!�����>�@9��A��\C.��                                    Bxp�׼  �          @�����
?�@G�BC 
���
>8Q�@b�\B�RC1Ǯ                                    Bxp��b  �          @������@*�H?޸RA�z�Ck�����?�@-p�A�G�C!0�                                    Bxp��  �          @����=q?��
@!G�A˙�C#:���=q>���@;�A��C0��                                    Bxp��  �          @����z�?��@S33B��C#���z�W
=@c33B��C6�3                                    Bxp�T  �          @����xQ�@"�\@Z=qBp�C�xQ�?Tz�@�p�B9�HC'�                                    Bxp� �  �          @�(���33?�
=@dz�B��C!�=��33�u@u�B%ffC7.                                    Bxp�/�  �          @�=q����?��H@[�B�C!�����;#�
@n�RB!{C6�                                    Bxp�>F  �          @�(�����@W�?�z�A���C:�����@{@K�B	�CE                                    Bxp�L�  �          @����Q�?���@c�
B{C&����Q��@j�HB��C;B�                                    Bxp�[�  �          @�(���z�?�{@p�A�=qC%:���z�>#�
@333A�z�C28R                                    Bxp�j8  �          @�(���p�?�
=@.{A�=qC&����p���@=p�A�Q�C5ff                                    Bxp�x�  �          @������@33?�33A��HC�����?��\@&ffA�C(}q                                    Bxp���  T          @�����@.�R?�(�Ak\)C�)���?���@!G�A�p�C��                                    Bxp��*  �          @�p���{@E�@
=A���C:���{?�  @^{B{C�                                    Bxp���  �          @����j�H@s33@ ��A�p�C��j�H@�@z�HB(�C��                                    Bxp��v  �          @���z=q@\)?�=qA���Cp��z=q@2�\@X��B��Cz�                                    Bxp��  �          @��R�c�
@��@hQ�B
=B���c�
@�@�p�B�� B��)                                    Bxp���  �          @�p��*=q@���@�HA�33B�R�*=q@A�@�p�B7��CO\                                    Bxp��h  �          @��R��33@j=q@�\A��C
8R��33@�\@j�HB��C�
                                    Bxp��  �          @���z�H@{�@ffA�{C���z�H@'
=@g
=B  Ck�                                    Bxp���  �          @���z�H@~{@��A�C���z�H@(Q�@j�HB\)C.                                    Bxp�Z  �          @���z�H@�Q�@33A�z�CW
�z�H@,��@fffB�Cn                                    Bxp�   �          @�
=��Q�@u�?5@��
C����Q�@G�@�A��CO\                                    Bxp�(�  �          @���z=q@{�?��A��RCٚ�z=q@,��@\(�Bp�Cs3                                    Bxp�7L  �          @�z��p��@�?L��@�(�CT{�p��@g�@+�A�\)C)                                    Bxp�E�  �          @��s33@��=#�
>��C@ �s33@�  @G�A�
=C}q                                    Bxp�T�  �          @�p����@x��?�Q�A<��C����@=p�@2�\A�\)C��                                    Bxp�c>  �          @�����(�@p  ?�\)A0  Ck���(�@7
=@*�HA�(�C�\                                    Bxp�q�  �          @�p��e�@�ff?@  @�C ���e�@i��@)��A�
=Ck�                                    Bxp���  �          @����L��@���u�(�B���L��@�@Q�A�z�B��                                    Bxp��0  �          @�z��y��@�Q�>�Q�@a�C� �y��@j=q@{A�(�C�H                                    Bxp���  �          @��
��@^{?O\)A33C����@/\)@G�A���C5�                                    Bxp��|  �          @������H@o\)?��AJffCL����H@1G�@4z�A噚C�                                     Bxp��"  �          @��R��33@l�;#�
����C0���33@XQ�?�G�Am�C�                                    Bxp���  �          @�{�~�R@c�
����ffC
.�~�R@�33�aG��(�C33                                    Bxp��n  �          @���=q@�?G�A�
C8R��=q?���?��A�33CE                                    Bxp��  
�          @�ff��{?�\)@p�A��HC 5���{?.{@4z�A�=qC,�
                                    Bxp���  �          @�z�����@!G�?��RAm�Cs3����?�=q@{A�{C"�)                                    Bxp�`  �          @�33��  @4z�?���A-C���  @�@  A��C�                                    Bxp�  �          @�����\)@N{?}p�A��C����\)@�@�A�CǮ                                    Bxp�!�  �          @����\)@n�R>\@p��C=q��\)@I��@G�A�=qC�)                                    Bxp�0R  �          @�33�s�
@�\)�&ff��G�C  �s�
@�=q?�=qAV�HC)                                    Bxp�>�  �          @�33�N�R@�=q�#�
��(�B����N�R@�=q@��A�p�B���                                    Bxp�M�  �          @�33��  @e?=p�@�=qCc���  @7�@�\A��Cz�                                    Bxp�\D  �          @��
����@r�\?�(�AA�C
s3����@4z�@3�
A��
C��                                    Bxp�j�  �          @�=q�Z�H@��?   @��B��=�Z�H@z�H@#�
A�{C
                                    Bxp�y�  �          @���QG�@�Q�=L��>�B��QG�@�
=@p�A��B���                                    Bxp��6  �          @����<��@�{���H���B�q�<��@��
?�A�z�B�(�                                    Bxp���  T          @������@r�\?�{AX��C	�����@0  @<(�A��
C}q                                    Bxp���  �          @�G��n{@\)?�A���C
=�n{@-p�@]p�B��C��                                    Bxp��(  �          @�G���=q@C�
@�
A�
=C�3��=q?�z�@]p�B{C�R                                    Bxp���  �          @��\�p  @i��@ ��A���CǮ�p  @ff@z=qB(��C�\                                    Bxp��t  �          @����1�@��?�(�A�=qB�3�1�@I��@uB)�Cn                                    Bxp��  �          @��H�  @�33?��A�G�B㙚�  @q�@s33B#\)B�\                                    Bxp���  �          @�����33@G�?0��@�=qCL���33@��@�\A��HC5�                                    Bxp��f  �          @����ff@{�?+�@�p�C����ff@K�@��A�ffC�)                                    Bxp�  �          @��\�X��@�z�B�\����B�L��X��@�ff?�(�A���C ��                                    Bxp��  �          @��H�7
=@��H����V{B�(��7
=@��?�ffA'
=B�p�                                    Bxp�)X  �          @������@|�Ϳ�{�XQ�C�R����@�(�?�R@��HCn                                    Bxp�7�  �          @���s�
@u��=q���C�\�s�
@�  >��?��
C�f                                    Bxp�F�  �          @�  �e�@vff�G����RCٚ�e�@�
=�k����C �                                    Bxp�UJ  �          @�G��y��@�  ���
�N=qCQ��y��@�(�?8Q�@�
=CY�                                    Bxp�c�  �          @����s33@��׿aG����CaH�s33@~{?��A2�RC�q                                    Bxp�r�  �          @�  �`��@�Q쾣�
�Mp�B��H�`��@�z�?�A��CQ�                                    Bxp��<  �          @����HQ�@�
=������B�\�HQ�@���>�@�Q�B�G�                                    Bxp���  �          @�p��C33@��Ϳ�p��?�B��C33@��?���A;�
B�                                    Bxp���  �          @��S33@�zᾣ�
�G
=B�  �S33@�
=@   A�{B�Ǯ                                    Bxp��.  �          @�\)����@���L�;�G�Cp�����@z�H@G�A�  C��                                    Bxp���  T          @��R�n{@��\��
=����C��n{@�  ?�\A���C33                                    Bxp��z  �          @�\)��ff@p  ?�\)A.�RC�
��ff@2�\@0  A�
=C��                                    Bxp��   �          @�z����@%�@0��A��C^����?}p�@i��B�HC'=q                                    Bxp���  �          @������@@��Aȣ�C�\����?:�H@EB
=C+#�                                    Bxp��l  �          @���(�?�(�?�z�A�Q�C!}q��(�?=p�@G�A�G�C+�\                                    Bxp�  �          @�{���?�@ ��A��
C&�
������@1G�A�G�C5�q                                    Bxp��  
�          @�Q���
=@\)?�p�A��\C�f��
=?��@333A�
=C'��                                    Bxp�"^  �          @������?���?�  Ap��C#�����?+�@�
A�G�C,�
                                    Bxp�1  �          @�����\)@�?L��@�z�C�H��\)?��H?���A�ffC!�)                                    Bxp�?�  �          @�����=q@C33?
=@���C����=q@=q?��HA��Cs3                                    Bxp�NP  �          @����@H��=�?�  C���@-p�?˅A�p�C�                                    Bxp�\�  �          @���33@{=�Q�?^�RC#���33@��?�  AK33CE                                    Bxp�k�  �          @�ff��ff@333>�\)@3�
Cu���ff@?�=qA�z�C��                                    Bxp�zB  �          @�
=��
=@L(��u��C���
=@:�H?��AT  C:�                                    Bxp���  �          @�����(�@R�\�u��C�{��(�@Vff?8Q�@�{C
                                    Bxp���  �          @������H@p  ?E�@�33C0����H@;�@p�A���C�3                                    Bxp��4  �          @�G��i��@���?O\)A ��C���i��@^�R@333A�\CQ�                                    Bxp���  �          @�����R@i��?0��@�33C�����R@8Q�@ffA�
=C!H                                    Bxp�À  �          @�
=��Q�@Fff?fffA33CL���Q�@�\@G�A�
=Cp�                                    Bxp��&  
�          @�ff���@e�?�
=A_\)CxR���@p�@=p�A�ffCG�                                    Bxp���  �          @��hQ�@�=q@  A�\)C�hQ�@   @z�HB&��C�                                     Bxp��r  �          @��R�E@��R?���A�33B��{�E@P��@vffB"ffCu�                                    Bxp��  �          @�����@^�R?��A"{C0�����@#�
@%�A�\)C33                                    Bxp��  �          @�  ��(�@\)?Y��A\)C	����(�@E@*�HAә�C�R                                    Bxp�d  �          @�  ��Q�@r�\?�\A�  C
T{��Q�@�R@XQ�B	��Cٚ                                    Bxp�*
  �          @�33��ff@7�@ ��A�
=C�
��ff?�ff@I��A�{C"�{                                    Bxp�8�  �          @ƸR��G�@�ff?���AG�
B����G�@��\@���B$�B�
=                                    Bxp�GV  �          @ƸR�u�@��?˅Apz�CxR�u�@J�H@c33B(�Cc�                                    Bxp�U�  �          @�����Q�@��?��AB�\CW
��Q�@L(�@P  A���C5�                                    Bxp�d�  �          @����Q�?���@33A�p�C#����Q�>��@$z�A�p�C/�                                     Bxp�sH  �          @���(�?J=q@��A���C,aH��(��\@�
A�33C7��                                    Bxp���  �          @�\)��ff?��R@�A��
C ��ff?
=@B�\A߮C.{                                    Bxp���  T          @�=q���@�@.�RAģ�C�3���?8Q�@aG�BC,��                                    Bxp��:  �          @�33��ff@��@Tz�A��HC+���ff>��H@�=qB��C.�{                                    Bxp���  �          @���
=?��@G
=Aߙ�C'G���
=�\@VffA�=qC7�\                                    Bxp���  �          @�p�����@��@XQ�A�Q�C!H����>�G�@��
B�C/{                                    Bxp��,  �          @ָR���@^{@C�
A�G�Cs3���?��@��B"G�C"^�                                    Bxp���  �          @�\)���@�ff@{A���CE���@3�
@�33B{C�q                                    Bxp��x  �          @�ff�p��@��?\ATz�B�Q��p��@u�@xQ�Bz�Cp�                                    Bxp��  �          @�Q����@��H@9��AӮC����@	��@���B0{C��                                    Bxp��  �          @�����  @�(�@�A��C�
��  @;�@�  B�
Cz�                                    Bxp�j  �          @�������?G�@�33BFG�C)�)���׿�@��\B8��CJ�                                    Bxp�#  �          @أ���=q?���@w
=Bz�C$���=q�0��@�G�B��C<!H                                    Bxp�1�  �          @�\)����?�=q@EA��
C"�����=��
@eB�RC3(�                                    Bxp�@\  �          @أ���z�?Ǯ@�ffB33C"Q���z�333@�B'p�C<(�                                    Bxp�O  T          @ڏ\��33@(�@q�B��C���33=L��@��
B;G�C3O\                                    Bxp�]�  �          @ڏ\���
@��@}p�B�C#����
<��
@��B,p�C3                                    Bxp�lN  �          @ٙ���ff@��@<��AУ�CJ=��ff@33@�B1\)CG�                                    Bxp�z�  �          @�33��ff@R�\@=p�A�C�f��ff?�z�@�ffB��C$�
                                    Bxp���  �          @�����\@p��@33A���C}q���\@��@hQ�B��C��                                    Bxp��@  �          @���z�@XQ�@o\)B33C���z�?���@�p�B7�C&��                                    Bxp���  �          @����@�
=@8��AŮC�����@p�@�
=B,=qC�\                                    Bxp���  �          @�����@�=q?�Q�A@Q�B��)����@z�H@xQ�B�
C�H                                    Bxp��2  �          @�(��u�@�?�{AZ=qB�p��u�@z�H@��\BCY�                                    Bxp���  �          @��H���@e@K�Aߙ�CE���?��@�G�B'�
C"5�                                    Bxp��~  �          @�\)���@Q�@=qA�G�C=q���?�z�@n{BQ�C"�f                                    Bxp��$  �          @ָR����?�{@(��A���C$������=�Q�@EA�33C3�                                    Bxp���  �          @�(���
=?�
=@AG�A���C%�R��
=���@U�A�C6�{                                    Bxp�p  T          @ҏ\����?���@5A�  C ������>�\)@[�A��C1\                                    Bxp�  �          @����@��@Y��A�=qC33��>Ǯ@�z�B33C/��                                    Bxp�*�  �          @������H@G�@r�\B��C  ���H=��
@��B#��C3{                                    Bxp�9b  �          @�����ff@j=q@\)A�
=C
޸��ff@�
@qG�B\)C��                                    Bxp�H  �          @��
��Q�@-p�@_\)A�(�C� ��Q�?(�@��
B=qC-c�                                    Bxp�V�  �          @�=q���@��@X��A�ffCW
���>�=q@�=qB��C10�                                    Bxp�eT  �          @�Q���p�@��@k�B��CY���p�=�@�=qB��C2��                                    Bxp�s�  �          @أ���Q�@�@K�A���C����Q�>\@y��B�
C0                                      Bxp���  �          @أ���(�@
=@UA�
=CW
��(�>�p�@�=qB33C0\                                    Bxp��F  �          @�=q���R@z�@`��A�{CB����R<�@�=qB�HC3�f                                    Bxp���  �          @ڏ\��(�?�p�@x��B�C$����(��+�@��B�RC;�                                    Bxp���  �          @ڏ\��>\)@��HB��C2������
=@g
=B 33CG�
                                    Bxp��8  �          @�(���33@U@ ��A��C��33?У�@vffB
{C#�                                    Bxp���  �          @ٙ����@��@p�A��RC�f���@*=q@�33B�CE                                    Bxp�ڄ  �          @����u�@���@G�A�{B���u�@U�@���B ��C
��                                    Bxp��*  �          @�33�qG�@�z�?�
=AD(�B���qG�@|(�@|(�B��C��                                    Bxp���  �          @�
=���\@��H�#�
����B������\@�(�@333A��HC ��                                    Bxp�v  �          @������\@��R?���A�RB�\���\@��@w
=BffC�                                    Bxp�  �          @���z�@�G�@33A�{B��\��z�@p  @��B  C	ٚ                                    Bxp�#�  �          @��
��@��\?�p�A��C
W
��@c33@`  A���C�{                                    Bxp�2h  �          @�\��\)@��H?��A%Cٚ��\)@E�@U�AظRC�R                                    Bxp�A  �          @�����@�  @p�A�Q�B��q���@u@��B!p�C	=q                                    Bxp�O�  �          @�33�j=q@�z�?�z�AP��B� �j=q@��R@�G�B��CT{                                    Bxp�^Z  �          @�  ��z�@�\)>��?�  Bӣ׿�z�@�\)@b�\A��B���                                    Bxp�m   �          @�����G��Y��@{�B!p�C>�\��G��.{@<��A�\)CR��                                    Bxp�{�  �          @��H��  @	��@/\)A��C33��  >�@\��B�C-�=                                    Bxp��L  �          @��H�333@w
=?^�RA%B��)�333@8Q�@-p�B  C#�                                    Bxp���  �          @�{�k�@:�H@*�HA�RC� �k�?�z�@q�B1�
C"p�                                    Bxp���  �          @��
��{@�\@?\)B�
C\��{>k�@g
=B"�C0�\                                    Bxp��>  �          @����?�ff@6ffA���C!8R��녽�@O\)B��C5s3                                    Bxp���  �          @�33��G�?+�@Z=qB�\C+&f��G���z�@QG�B�CC#�                                    Bxp�ӊ  �          @�G���{>�G�@�33B$�C.����{���H@p  B�CH�                                    Bxp��0  �          @�����33>.{@�
=B*
=C1ٚ��33�   @n{B(�CK��                                    Bxp���  �          @������\>k�@uB&=qC0�����\��G�@Z�HB��CJ)                                    Bxp��|  �          @�(����\?
=q@h��B�C-=q���\���@Z=qBCD�f                                    Bxp�"  �          @�����ff>�Q�@VffB��C/�
��ff��33@Dz�B 
=CD��                                    Bxp��  T          @�  ��
=>�ff@Mp�B=qC.����
=��G�@>�RA�33CB�                                    Bxp�+n  �          @��H���H=�\)@N{B��C35����H�˅@333A�z�CF5�                                    Bxp�:  �          @����;��@i��BQ�C5�������   @C33A��RCKE                                    Bxp�H�  �          @�Q���=q���@fffBG�CA���=q�/\)@#�
A�
=CTc�                                    Bxp�W`  �          @��
���H?Ǯ@R�\BC!8R���H��33@hQ�B��C8\)                                    Bxp�f  �          @�Q���\)@*�H@FffA�p�C�=��\)?333@���B p�C+�=                                    Bxp�t�  �          @\�~�R@���@{A�G�C�f�~�R@�
@~{B#33C��                                    Bxp��R  �          @��R�j=q@��?�Q�A���C���j=q@(��@uB!��C5�                                    Bxp���  �          @����[�@j�H@A�A�z�C
�[�?˅@�\)BG\)C#�                                    Bxp���  �          @�zῬ��@�(�?�(�AG
=B�33����@~�R@tz�B(�B�z�                                    Bxp��D  �          @�Q�0��@�=q?�\)A,��B����0��@��@|��B&�B��)                                    Bxp���  �          @�z���@�ff@��A�
=Bݔ{���@E�@��BL�B���                                    Bxp�̐  �          @���r�\?ٙ�@G�B�C�{�r�\���@c33B,p�C6L�                                    Bxp��6  T          @��
�z�H@��
@�A߅B�.�z�H@!G�@�G�Bg=qB�ff                                    Bxp���  �          @�z��@  @HQ�@G
=Bz�CǮ�@  ?�=q@���BU=qC E                                    Bxp���  �          @���%@`��@33Aə�B��R�%?���@dz�B?G�C�                                    Bxp�(  �          @�{�+�@J�H@	��A��HC:��+�?���@^�RB@G�C#�                                    Bxp��  �          @�  �?\)@G�@E�Bz�C�\�?\)>�33@s�
BN�RC-^�                                    Bxp�$t  �          @�p��Mp�?�(�@a�B,��C^��Mp��\)@�G�BN=qC6z�                                    Bxp�3  �          @��H�S�
?Ǯ@aG�B/�C���S�
���H@tz�BB��C<p�                                    Bxp�A�  �          @��j�H@�@S�
Bp�CL��j�H>\)@}p�B<�
C1                                    Bxp�Pf  �          @��\�,��@n{?�
=A��\B��,��@�@I��B#�RC	��                                    Bxp�_  �          @�z��p�@�  ?���A��B���p�@Fff@e�B0�B�G�                                    Bxp�m�  �          @�����@���?�\)A�\)B�=q��@p�@k�B=�Cn                                    Bxp�|X  �          @�����H@��R?�Q�A\��B��)��H@>{@L��B�\C(�                                    Bxp���  �          @�  ���@���?+�@�  B���@S33@2�\B	z�B�                                      Bxp���  �          @�33�Fff@.�R@z�AԸRC
���Fff?�G�@L(�B.p�C�3                                    Bxp��J  �          @������?�\@��A��C-\)����5@	��A��C=J=                                    Bxp���  �          @�p���녿^�R@
�HA�(�C>����녿�\)?�33A|��CJT{                                    Bxp�Ŗ  �          @�������.{@��A�C6&f������?���A�G�CEu�                                    Bxp��<  �          @�\)��Q�?Q�?��A�\)C)�3��Q쾔z�@�\A�z�C7��                                    Bxp���  �          @��
���H�333?�z�A�G�C=.���H���H?�ffAN=qCF�H                                    Bxp��  �          @���1녿���?�(�A�{CQ���1��   >L��@W�CW�R                                    Bxp� .  �          @����{@��\?�{AD(�B��R��{@p  @fffB.��B�p�                                    Bxp��  �          @��׿}p�@�  @33A�\)Bˣ׿}p�@>{@�\)BU��B��                                    Bxp�z  �          @�{�fff@�G�?���A�z�B���fff@Vff@�{BI��B��                                    Bxp�,   �          @�녾�Q�@���@ ��A�=qB�녾�Q�@U@��BR�B�33                                    Bxp�:�  �          @�=q<��
@���?�Q�A��
B���<��
@u@�33BB=qB�k�                                    Bxp�Il  �          @�=q�.{@��
?�Q�Af=qB�aH�.{@���@�B7�HB��3                                    Bxp�X  �          @���?�\@�\)?޸RA�ffB�L�?�\@p��@��BC�B�k�                                    Bxp�f�  �          @�G�?��\@�=q@   A�Q�B�B�?��\@^�R@�  BLp�B�G�                                    Bxp�u^  �          @�{?�  @�z�?�(�A�G�B���?�  @Vff@�(�BK�B���                                    Bxp��  �          @���5�@H��@ffAӮC��5�?Ǯ@\(�B:�
C
                                    Bxp���  �          @��C�
@S33@�RA�\)C���C�
?�  @vffBA��C�
                                    Bxp��P  �          @��R�$z�@b�\@+�A�(�B��H�$z�?���@���BWffC                                    Bxp���  �          @�(��"�\@\��@1G�Bz�B��
�"�\?�p�@�p�B[(�C޸                                    Bxp���  �          @���   @Y��@>�RB(�B��)�   ?�=q@�=qBs�HCk�                                    Bxp��B  �          @�����\@��
@$z�A�ffB�8R���\@
=@��Bp��B➸                                    Bxp���  �          @��\�J=q@���@A��HB�p��J=q@ ��@���BjG�B�{                                    Bxp��  �          @�\)=���@��
@�A�  B�#�=���@Q�@~�RBl�B�
=                                    Bxp��4  �          @�G��Tz�@���?���A�p�B�G��Tz�@��@l(�B[��B�k�                                    Bxp��  �          @�z��ff@S�
@FffB+��Bã׾�ff?�@�(�B���B�W
                                    Bxp��  �          @��
�B�\@n{@(Q�B
�\B�{�B�\?��
@�{B�� B�Q�                                    Bxp�%&  �          @����\@n�R@,(�B33Bî��\?�  @�Q�B���BԽq                                    Bxp�3�  �          @���#�
@|(�@H��B��B���#�
?�Q�@�  B���B��                                     Bxp�Br  �          @����
=@��
@�A�33B�(���
=@AG�@��
BU�HBޞ�                                    Bxp�Q  �          @��\��@�=q?�\)A�\)B��Ϳ�@\(�@��BE=qBƽq                                    Bxp�_�  �          @�G��p��@�33@  Ȁ\B�
=�p��@.{@��HBa��B��                                    Bxp�nd  �          @��
>�G�@�=q@  AǅB���>�G�@9��@��RBbB��H                                    Bxp�}
  �          @��\��(�@���@%�A�p�B߸R��(�@G�@�ffBe��B�B�                                    Bxp���  �          @��\��33?��\?�Q�A��C#����33=L��@z�A�Q�C3J=                                    Bxp��V  �          @��\��z�?�Q�@��A�\C �\��z�#�
@2�\B��C4��                                    Bxp���  �          @�Q���?�=q@�Aأ�CG���>�@1G�BG�C28R                                    Bxp���  �          @��\��Q�?�p�?�G�A���C!����Q�>��
@�A��C/��                                    Bxp��H  �          @�G���(�?xQ�?��HA��RC(.��(����
?��HA���C5�                                    Bxp���  �          @����?�{?��HA���C 
=��>�@�
A���C.!H                                    Bxp��  �          @��H���@	��?�Q�A��RC#����?k�@$z�A�\)C'��                                    Bxp��:  �          @�33��p�?c�
?�\)A���C)=q��p���G�?���A��\C5Y�                                    Bxp� �  �          @���u�@��?���A�Cs3�u�?aG�@7
=B{C')                                    Bxp��  �          @������\?p��@33A�G�C'����\��(�@��A��C9�H                                    Bxp�,  �          @��H��=q���
?���A���C7����=q���?��AAp�C@5�                                    Bxp�,�  �          @����
=��ff?�A�{C9z���
=��{?���Ap��CD!H                                    Bxp�;x  �          @���|(��>{>L��@G�CY��|(��'���z���33CU�)                                    Bxp�J  �          @���s33�;���Q쿋�CY���s33�(�������CT��                                    Bxp�X�  �          @�z������G�?J=qA ��C@�q������\>\)?�(�CDJ=                                    Bxp�gj  �          @��\�~{@�\?���A��C
=�~{?���@��A�G�C#&f                                    Bxp�v  �          @�����\)@�
?�Q�A�G�Ck���\)?�
=@��A�
=C$c�                                    Bxp���  �          @��H��\)?�=q?�  ADz�C&p���\)>\?�A���C/(�                                    Bxp��\  T          @����
=?Y���#�
����C)Ǯ��
=?L��>��
@w
=C*Y�                                    Bxp��  �          @��
����?�{��G���C������?˅>�@��C {                                    Bxp���  �          @������H?�ff?�R@�C ff���H?z�H?���A���C'L�                                    Bxp��N  �          @���>{@X��?��\AL(�C:��>{@z�@*�HBQ�C                                      Bxp���  �          @�p��h��?��@?\)B��C =q�h�þ��H@N�RB%(�C;�                                    Bxp�ܚ  �          @�z��(��@7�@{B��C���(��?���@g�BN�HCL�                                    Bxp��@  �          @���@|��@G�A�z�B�ff��@  @tz�BL  C.                                    Bxp���  �          @����
@�z�?��A�p�B����
@\)@s33BL�B�{                                    Bxp��  �          @�(���  @tz�@&ffB\)B��)��  ?�@��Bs{C33                                    Bxp�2  �          @��\���
@�{@	��Aә�BϞ����
@��@���Be(�B♚                                    Bxp�%�  �          @��H��=q@�p�?�\A��B�aH��=q@%�@n�RBPffB�{                                    Bxp�4~  �          @�=q��{@���?޸RA�p�B��῎{@,��@qG�BQ�B�                                    Bxp�C$  �          @�����Q�@�(�?ٙ�A���B�aH��Q�@%�@i��BG(�B�Q�                                    Bxp�Q�  �          @�=q� ��@o\)?�(�A���B���� ��@�@P  B-
=C�)                                    Bxp�`p  �          @�녿���@�(�?˅A�33B�.����@(Q�@c�
B?�\B�{                                    Bxp�o  �          @�녿޸R@�Q�?�{A�=qB�aH�޸R@7
=@[�B6�HB�u�                                    Bxp�}�  �          @�녿(�@�{>�33@�Q�B��Ϳ(�@qG�@3�
BffB�\)                                    Bxp��b  �          @������@�z�?�p�Aj=qB�����@5@P��B(B��                                    Bxp��  �          @�ff�2�\@j�H?�(�A�B���2�\@
�H@Z�HB0�C{                                    Bxp���  �          @�p����@qG�?�p�A�{B�.���@�@l(�BB�
C8R                                    Bxp��T  �          @����(�@p  ?�z�A��RB�33�(�@G�@Z=qB6�C	�                                    Bxp���  �          @��H�E�@G�@�
AɮC��E�?\@Z=qB3=qC�R                                    Bxp�ՠ  �          @��\�N�R@(��@�A��C�3�N�R?u@Z=qB5��C#xR                                    Bxp��F  �          @�(��N�R@%�@!�A�p�CaH�N�R?Q�@aG�B:33C%                                    Bxp���  �          @���9��@R�\@ffȀ\Ch��9��?�33@a�B:�Ch�                                    Bxp��  �          @������@{�?��A�
=B�����@�@`  B6�\C�                                    Bxp�8  �          @�=q�'�@���?�\)A�33B�(��'�@!G�@aG�B0�C�                                    Bxp��  �          @����@�
=?��A���B���@.�R@dz�B1ffC�)                                    Bxp�-�  �          @�  �,(�@y��?ǮA��B�8R�,(�@��@Z=qB,�\C	�                                    Bxp�<*  �          @�G��'�@�  ?�=qA�  B�z��'�@!�@^�RB.�C{                                    Bxp�J�  �          @��
�HQ�@s33?�ffA@��C���HQ�@(��@;�B=qC�                                    Bxp�Yv  �          @����C33@|��?   @���B�L��C33@C�
@#33A�z�C��                                    Bxp�h  �          @�p��(�@����#�
��33B���(�@mp�@�A֣�B�
=                                    Bxp�v�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxp��h              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxp��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxp���  
�          @��׿��@�Q�?uA:�\B�LͿ��@Q�@O\)B+�B�8R                                    Bxp��Z  �          @��H��z�@z=q@(Q�B  Bۮ��z�?��@�=qBu��B��3                                    Bxp��   �          @����@\��@C33B��B��H��?�G�@�Bn33C�                                    Bxp�Φ  �          @����Q�@L(�@P��B!p�B�p��Q�?n{@�
=Bz  C^�                                    Bxp��L  �          @�p��ff@)��@XQ�B.�RC��ff>��@���Bsz�C*\                                    Bxp���  �          @���.�R@��@Z=qB1�
C�H�.�R��@���B_\)C4��                                    Bxp���  �          @�p��=p�@�R@Mp�B#ffC  �=p�>��@z=qBS�C1�                                    Bxp�	>  �          @����@  @
=q@9��BG�CQ��@  >��@fffBHQ�C/#�                                    Bxp��  �          @�z��fff?h��@0��B\)C%ٚ�fff�8Q�@3�
B�RC?W
                                    Bxp�&�  �          @�Q��Tz�?�(�@=p�B�
C��TzᾸQ�@R�\B2=qC:+�                                    Bxp�50  �          @�\)���@B�\@EBffC ����?fff@��Bk��CaH                                    Bxp�C�  �          @��p�@Z=q@!�A�{B���p�?�G�@~{BW�
C�=                                    Bxp�R|  �          @����$z�@^�R@Q�A��B���$z�?��
@j�HBF\)CO\                                    Bxp�a"  �          @�z��!�@^�R@
�HA�B�\�!�?�G�@l��BH��C+�                                    Bxp�o�  �          @�ff�(�@e@A��B�p��(�?�G�@z=qBQ��C:�                                    Bxp�~n  �          @�p��=q@g
=@
�HA��HB�z��=q?�{@qG�BLffCE                                    Bxp��  �          @��׿���@s33@,(�B��B��
����?�  @��Bm�
C��                                    Bxp���  �          @�����\@�Q�@�RA��
B�׿�\@�\@�  BfQ�C��                                    Bxp��`  �          @�  ��@i��@%A�z�B�=��?�
=@���Ba�CT{                                    Bxp��  �          @�ff�-p�@Tz�@{A�(�C8R�-p�?��H@w�BN=qC�R                                    Bxp�Ǭ  �          @�녿�
=@tz�@33A�p�B�ff��
=?�(�@�  Bdz�Cff                                    Bxp��R  �          @����ff@~�R@��AՅB�q��ff@
=q@�  B[\)C�\                                    Bxp���  �          @��\����@�@(�A��B�녿���@@�33BX�B���                                    Bxp��  �          @�  �   @�33@ ��A�{B�#��   @
=@y��BN33CL�                                    Bxp�D  �          @�z��Q�@�=q?���A�(�B����Q�@�@p��BIp�C �H                                    Bxp��  �          @�{���@|��@��AڸRB��
���@ff@�G�B\G�C�                                     Bxp��  
�          @��
���@s�
@
=A�ffB����?�
=@�G�BaG�C8R                                    Bxp�.6  �          @��\���R@fff@\)A�33B��f���R?�Q�@�G�Bd�C�q                                    Bxp�<�  �          @�G���p�@s�
@�AΣ�B���p�@�@s�
BS��C��                                    Bxp�K�  �          @�  ��{@��R?���AV�HB����{@<��@L��B*(�B�z�                                    Bxp�Z(  �          @�ff��\@z=q?aG�A/�B����\@4z�@6ffB��C
                                    Bxp�h�  �          @�(��.�R@`  ?��A�{B��
�.�R@\)@>�RB ��C�                                    Bxp�wt  �          @�����
@�  ?�z�A��\B��f���
@'
=@VffB:B��{                                    Bxp��  �          @����{@y��?�Q�A|  B����{@(��@FffB4�\B��                                    Bxp���  �          @�p����R@u?Y��A2�HB����R@1G�@2�\BQ�B�\)                                    Bxp��f  �          @�Q��!G�@`��?��Ac
=B�Q��!G�@�@4z�BC�\                                    Bxp��  �          @�(��(Q�@]p�?ǮA�Q�B�z��(Q�@z�@L(�B.ffC�
                                    Bxp���  �          @�z��6ff@XQ�?��A�  C0��6ff@@@  B!33C�=                                    Bxp��X  �          @���,��@c�
?��\A~ffB�aH�,��@�
@?\)B z�C}q                                    Bxp���  
�          @�Q��7
=@j=q?z�HA=��B��)�7
=@#�
@333B��C
(�                                    Bxp��  �          @����2�\@r�\?!G�@�33B��R�2�\@6ff@%�Bp�Cc�                                    Bxp��J  �          @�ff�-p�@u�>.{@�
B��=�-p�@HQ�@p�A���Cٚ                                    Bxp�	�  �          @���   @k�>�@ƸRB�u��   @5�@��BffCs3                                    Bxp��  �          @�G��Ǯ@\(�?�33A��HB�Ǯ�Ǯ@��@B�\BC��B�=q                                    Bxp�'<  �          @��׿�@Z=q?z�HAc�B�녿�@@*=qB(
=B�k�                                    Bxp�5�  �          @�33��R@U?h��AL  B�Q���R@z�@$z�B�C�=                                    Bxp�D�  �          @��ͽ���@XQ�@Q�B�B�aH����?Ǯ@u�B���B�Q�                                    Bxp�S.  �          @���\)@`��@{B(�B�� �\)?�  @p��B��HB�                                      Bxp�a�  �          @���W
=@fff@z�A���B����W
=?�z�@k�By�HB�p�                                    Bxp�pz  �          @���{@g
=@z�A�(�B��R��{?�@l(�Bx�HB�                                    Bxp�   �          @�{��33@l(�?��HA�z�B݊=��33@��@@��B;p�B�p�                                    Bxp���  �          @��H��G�@a�?�{A�  B�LͿ�G�@\)@C33BB
=B�                                    Bxp��l  �          @{���\@Q�?��HA�  B�=��\@
=@2�\B5C�                                    Bxp��  �          @��ÿ�
=@XQ�?�{A�z�B��ÿ�
=@
=@>{B>�C �{                                    Bxp���  �          @��\���@L��?ٙ�A��B���?�\@J=qBM�C\)                                    Bxp��^  �          @��R�	��@Q�?�{A�\)B���	��@G�@:�HB2��C�q                                    Bxp��  �          @�\)��  @��׾���`��Bօ��  @aG�?���AܸRB�
=                                    Bxp��  �          @�p��  @Tz�?Y��A?�B�W
�  @�@ ��B�
C��                                    Bxp��P  �          @��
�Q�@XQ�?z�HA]��B�aH�Q�@�
@)��B �C��                                    Bxp��  �          @���33@I��?��Am�B�G��33@@$z�B\)C	��                                    Bxp��  �          @�  ��=q@U�?��A�  B��R��=q@p�@-p�B-\)C�f                                    Bxp� B  �          @�p��  @R�\?���A|  B����  @
�H@.�RB$p�C�                                    Bxp�.�  �          @�Q��(�@c33?��\A�Q�B����(�@�
@>�RB1�HC�                                    Bxp�=�  �          @�녿��H@c33?���A�  B����H@��@H��B;�C�q                                    Bxp�L4  �          @��׿�  @z=q?�=qAl(�B׏\��  @-p�@AG�B5\)B�                                    Bxp�Z�  �          @�G���  @s�
?�G�A���B�
=��  @!�@G�B:�B�p�                                    Bxp�i�  �          @��ÿ�p�@z�H?O\)A0(�B�uÿ�p�@7
=@333B#��B���                                    Bxp�x&  �          @�G����\@���?k�AG33B�ff���\@:=q@=p�B/33Bڽq                                    Bxp���  �          @�����@~{?E�A&�\B�녿��@;�@2�\B$(�B�k�                                    Bxp��r  �          @��R�xQ�@{�?���Ar{B�Ǯ�xQ�@.{@B�\B9�HB�W
                                    Bxp��  �          @���ff@s33?0��AB�z��ff@4z�@(Q�B��B홚                                    Bxp���  �          @��ٙ�@h��?��\Ac33B�{�ٙ�@!G�@4z�B+ffB�\                                    Bxp��d  �          @���@S�
?
=A��B�#���@{@�B�C��                                    Bxp��
  
c          @�(��?\)@5�>.{@��C���?\)@�\?�A��C�\                                    Bxp�ް  �          @��Q�@#�
    �#�
C  �Q�@
=q?���A���C�H                                    Bxp��V  �          @�  �7
=@G
=>�
=@�p�C���7
=@��@�A�Q�C\                                    Bxp���  �          @�Q��U@%���=q�l(�CL��U@z�?�z�A~�HC33                                    Bxp�
�  �          @����^{@ff�
=q��C�f�^{@G�?O\)A0��C�=                                    Bxp�H  �          @�Q��^{@{�k��C33C���^{@��?��Aw�C��                                    Bxp�'�  �          @�G��h��@��B�\�#\)C�\�h��@(�?   @�
=C�                                    Bxp�6�  �          @����X��@�R�G��(  C���X��@ ��?.{AffCz�                                    Bxp�E:  �          @�p��u�?�Q쿃�
�XQ�C{�u�@(�>8Q�@�C.                                    Bxp�S�  
�          @��
�L��@:=q���Ϳ�=qC	��L��@ ��?�p�A�=qC��                                    Bxp�b�  �          @�33�P  @6ff�k��?\)C
���P  @!G�?��A���C#�                                    Bxp�q,  T          @�p��Tz�@7
=>�\)@g
=CE�Tz�@��?�\A�=qC��                                    Bxp��  
�          @�p��`  @+�>W
=@0  C���`  @��?�\)A�  C�=                                    Bxp��x  "          @�(��[�@.{�#�
��C�\�[�@33?��HA�G�C&f                                    Bxp��  �          @���W�@0  �L�Ϳ+�CǮ�W�@ff?�Q�A�{C&f                                    Bxp���  �          @��R�]p�@1G���(���Q�CE�]p�@$z�?���Aj�HCc�                                    Bxp��j  �          @�
=�Y��@1G��Q��)�C�
�Y��@1�?J=qA$  C                                    Bxp��  �          @�G��Y��@2�\����hz�C�{�Y��@>{?
=q@�G�C
ٚ                                    Bxp�׶  �          @�\)�QG�@/\)������C��QG�@A�>�{@��\C	.                                    Bxp��\  �          @��Mp�@%��˅���C8R�Mp�@A�=#�
>�C��                                    Bxp��  �          @���H��@1녿������C
z��H��@C�
>\@�{C��                                    Bxp��  �          @�(��K�@+���{��
=C�H�K�@?\)>�z�@q�C�                                     Bxp�N  T          @�z��Q�@"�\��z����CL��Q�@9��>.{@�C
�
                                    Bxp� �  �          @����Q�@%��������C޸�Q�@:�H>W
=@,(�C
T{                                    Bxp�/�  ~          @�(��L��@(�ÿ�z���Cc��L��@?\)>k�@@��C�                                    